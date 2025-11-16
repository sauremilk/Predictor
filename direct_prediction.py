#!/usr/bin/env python3
"""
Direkte Modell-Nutzung ohne API - lädt Pipeline direkt aus Joblib
Am robustesten: Keine Netzwerk-Dependencies, kein Server nötig
"""
import joblib
import pandas as pd
import json
from pathlib import Path

# Modell laden
MODEL_PATH = Path(__file__).parent / "Vorhersage-Modell" / "models" / "baseline_pipeline_final.joblib"
print(f"📦 Lade Modell von: {MODEL_PATH}")
pipeline = joblib.load(MODEL_PATH)

# Game State definieren
game_state = {
    "zone_phase": "late",
    "zone_index": 7,
    "alive_players": 15,
    "teammates_alive": 2,
    "storm_edge_dist": 50.0,
    "mats_total": 150,
    "surge_above": 5,
    "height_status": "low",
    "position_type": "corner",
    # Optional: Outcome-Felder (werden für Prediction ignoriert, aber Pipeline erwartet sie)
    "outcome_placement": 0,
    "outcome_alive_time": 0
}

print(f"\n🎮 Game State:")
for key, value in game_state.items():
    print(f"  {key:20s}: {value}")

# Als DataFrame (Pipeline erwartet DataFrame-Input)
df = pd.DataFrame([game_state])

# Prediction
predicted_class = pipeline.predict(df)[0]
probabilities = pipeline.predict_proba(df)[0]
class_names = pipeline.classes_

print(f"\n✅ Prediction:")
print(f"  Empfohlener Call: {predicted_class}")
print(f"  Confidence: {probabilities.max():.2%}")

print(f"\n📊 Alle Wahrscheinlichkeiten:")
for name, prob in sorted(zip(class_names, probabilities), key=lambda x: x[1], reverse=True):
    if prob > 0.01:  # Nur relevante anzeigen
        bar = "█" * int(prob * 50)
        print(f"  {name:25s}: {prob:6.2%} {bar}")

# Als JSON exportieren (für weitere Verarbeitung)
result = {
    "predicted_call": predicted_class,
    "confidence": float(probabilities.max()),
    "probabilities": {name: float(prob) for name, prob in zip(class_names, probabilities) if prob > 0.01}
}

print(f"\n💾 JSON Output:")
print(json.dumps(result, indent=2))
