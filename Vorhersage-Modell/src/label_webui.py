import os
import json
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd


DATA_CSV = os.environ.get("CALL_STATES_TO_LABEL", "Vorhersage-Modell/__pycache__/processed/call_states_synth_large.csv")
METADATA_PATH = os.environ.get("IMAGE_METADATA", "Vorhersage-Modell/__pycache__/data/dummy_screenshots/metadata.jsonl")
LABELED_CSV = os.environ.get("CALL_STATES_LABELED", "Vorhersage-Modell/__pycache__/processed/call_states_labeled.csv")

BEST_CALL_CLASSES: List[str] = [
    "stick_deadside",
    "play_frontside",
    "take_height",
    "stabilize_box",
    "look_for_refresh",
    "drop_low",
]


class LabelRequest(BaseModel):
    match_id: str
    frame_id: int
    best_call: str


def load_metadata(path: str) -> pd.DataFrame:
    rows = []
    if os.path.exists(path):
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    if not rows:
        return pd.DataFrame(columns=["match_id", "frame_id", "image_path"])
    df = pd.DataFrame(rows)
    # Ensure types
    df["match_id"] = df["match_id"].astype(str)
    df["frame_id"] = df["frame_id"].astype(int)
    return df


def load_call_states(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Call states CSV not found: {path}")
    df = pd.read_csv(path)
    if "match_id" not in df.columns or "frame_id" not in df.columns:
        # Fallback: create dummy IDs if not present
        df["match_id"] = "demo_000"
        df["frame_id"] = range(len(df))
    df["match_id"] = df["match_id"].astype(str)
    df["frame_id"] = df["frame_id"].astype(int)
    return df


def load_labeled(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=["match_id", "frame_id", "best_call"])
    df = pd.read_csv(path)
    df["match_id"] = df["match_id"].astype(str)
    df["frame_id"] = df["frame_id"].astype(int)
    return df


def save_labeled(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


app = FastAPI(title="Best Call Label WebUI")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    # Minimal HTML with inline JS
    html = """<!DOCTYPE html>
<html>
<head>
<meta charset=\"utf-8\" />
<title>Best Call Labeling</title>
<style>
body { font-family: sans-serif; margin: 1rem; }
#container { display: flex; gap: 1rem; }
#image-container img { max-width: 640px; border: 1px solid #ccc; }
#features { max-width: 400px; font-size: 14px; }
button.label-btn { margin: 0.25rem; padding: 0.5rem 0.75rem; }
#status { margin-top: 1rem; color: #555; }
</style>
</head>
<body>
<h1>Best Call Labeling</h1>
<div id=\"status\">Lade Szene...</div>
<div id=\"container\">
  <div id=\"image-container\">
    <img id=\"scene-image\" src=\"\" alt=\"Scene\" />
  </div>
  <div id=\"features\">
    <h3>Features</h3>
    <pre id=\"feature-json\"></pre>
    <h3>Label</h3>
    <div id=\"buttons\"></div>
  </div>
</div>
<script>
const BEST_CALL_CLASSES = [
  \"stick_deadside\",
  \"play_frontside\",
  \"take_height\",
  \"stabilize_box\",
  \"look_for_refresh\",
  \"drop_low\"
];

let currentSample = null;

async function fetchNextSample() {
  const res = await fetch('/next_sample');
  if (!res.ok) {
    document.getElementById('status').innerText = 'Keine weiteren Szenen oder Fehler beim Laden.';
    return;
  }
  const data = await res.json();
  currentSample = data;
  document.getElementById('status').innerText = `Match ${data.match_id}, Frame ${data.frame_id}`;
  document.getElementById('scene-image').src = data.image_url;
  document.getElementById('feature-json').innerText = JSON.stringify(data.features, null, 2);
}

async function sendLabel(label) {
  if (!currentSample) return;
  const payload = {
    match_id: currentSample.match_id,
    frame_id: currentSample.frame_id,
    best_call: label
  };
  const res = await fetch('/label', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });
  if (!res.ok) {
    alert('Fehler beim Speichern des Labels');
    return;
  }
  await fetchNextSample();
}

function initButtons() {
  const container = document.getElementById('buttons');
  BEST_CALL_CLASSES.forEach(c => {
    const btn = document.createElement('button');
    btn.innerText = c;
    btn.className = 'label-btn';
    btn.onclick = () => sendLabel(c);
    container.appendChild(btn);
  });
  const skip = document.createElement('button');
  skip.innerText = 'Skip';
  skip.className = 'label-btn';
  skip.onclick = () => fetchNextSample();
  container.appendChild(skip);
}

initButtons();
fetchNextSample();
</script>
</body>
</html>
"""
    return HTMLResponse(content=html)


@app.get("/next_sample")
async def next_sample() -> JSONResponse:
    calls = load_call_states(DATA_CSV)
    meta = load_metadata(METADATA_PATH)
    labeled = load_labeled(LABELED_CSV)

    # Merge to know which ones are already labeled
    merged = calls.merge(meta, on=["match_id", "frame_id"], how="left")
    merged = merged.merge(labeled, on=["match_id", "frame_id"], how="left", suffixes=("", "_labeled"))

    # Unlabeled = best_call missing in labeled df
    candidates = merged[merged["best_call_labeled"].isna()]
    if candidates.empty:
        raise HTTPException(status_code=404, detail="No unlabeled samples left")

    row = candidates.iloc[0]
    image_path = row.get("image_path")
    if not isinstance(image_path, str) or not os.path.exists(image_path):
        # If image is missing, skip this one and try next
        # To avoid recursion depth, filter here
        candidates = candidates[candidates["image_path"].notna()]
        candidates = candidates[candidates["image_path"].apply(lambda p: isinstance(p, str) and os.path.exists(p))]
        if candidates.empty:
            raise HTTPException(status_code=404, detail="No unlabeled samples with images found")
        row = candidates.iloc[0]
        image_path = row["image_path"]

    # Build image URL via /image endpoint
    image_url = f"/image?path={image_path}"

    # Features: drop label columns
    feature_cols = [c for c in calls.columns if c not in ["best_call"]]
    features = {col: row[col] for col in feature_cols if col in row}

    return JSONResponse({
        "match_id": row["match_id"],
        "frame_id": int(row["frame_id"]),
        "image_url": image_url,
        "features": features,
    })


@app.post("/label")
async def label_scene(req: LabelRequest) -> JSONResponse:
    if req.best_call not in BEST_CALL_CLASSES:
        raise HTTPException(status_code=400, detail="Invalid best_call class")

    labeled = load_labeled(LABELED_CSV)

    # Upsert row
    mask = (labeled["match_id"] == req.match_id) & (labeled["frame_id"] == req.frame_id)
    if mask.any():
        labeled.loc[mask, "best_call"] = req.best_call
    else:
        labeled = pd.concat([
            labeled,
            pd.DataFrame({
                "match_id": [req.match_id],
                "frame_id": [req.frame_id],
                "best_call": [req.best_call],
            }),
        ], ignore_index=True)

    save_labeled(labeled, LABELED_CSV)
    return JSONResponse({"status": "ok"})


@app.get("/image")
async def serve_image(path: str):
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Image not found: {path}")
    return FileResponse(path)
