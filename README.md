# Predictor — Baseline für `best_call`

Dieses Repo enthält ein Scaffold für ein Mehrklassen-Klassifikationsproblem (Ziel: `best_call`).

Schnellstart (CPU):

```bash
python3 src/train_best_call_baseline.py --data data/call_states_demo.csv
```

Dateien:
- `data/call_states_demo.csv` — kleiner synthetischer Beispiel-Datensatz
- `data/processed/call_states_synth_large.csv` — größerer synthetischer Datensatz
- `src/train_best_call_baseline.py` — Trainingsskript (Preprocessing, CV, GridSearchCV)
- `src/generate_synth_large.py` — Synthesizer für größeren Datensatz
- `src/export_to_onnx.py` — Export-Helper (exportiert Klassifier nach ONNX)
- `src/predict_best_call.py` — Inference helper (preprocess in Python + ONNX classifier)
- `models/` — gespeicherte Pipeline/Modelle und Reports
- `notebooks/baseline_best_call.ipynb` — kurzes Notebook mit Anleitung
- `notebooks/eval_best_call_baseline.ipynb` — Evaluations-Notebook

Zielmetrik: Accuracy (Primär), macro F1 (zusätzlich).

Deployment
----------

ONNX export:
- Der Klassifizierer (RandomForest) wurde als `models/best_call_baseline.onnx` exportiert.
- Das ONNX-Modell erwartet einen preprocessed Float-Vektor (Output der Preprocessor-Stage `pre`).
- Für Inferenz wendet das Inference-Skript (`src/predict_best_call.py`) die Python-Preprocessing-Pipeline
	aus `models/baseline_pipeline_final.joblib` an und übergibt den resultierenden Float-Vektor an das ONNX-Modell.

Quick inference (Beispiel):

```bash
python3 -m src.predict_best_call --input example.json
```

Wo `example.json` eine einzelne Game-State-Zeile als JSON-Dict enthält (Feldnamen siehe `data/processed/call_states_synth_large.csv`).

Der Ablauf:
- `src/predict_best_call.py` lädt das Joblib-Pipeline (`pre` + `clf`), wendet `pre.transform` an und ruft dann die ONNX-Klasse
	via `onnxruntime` auf, um `predict` und `predict_proba` zu erhalten.

Hinweis: Der ONNX-Export enthält nur den Klassifizierer (Classifier-only). Die Preprocessing-Logik bleibt in Python,
da die vollständige Pipeline mit Imputern/OneHotEncoder einige Konverter-Einschränkungen aufweist.

Batch predictions
-----------------

Ein kleines E2E-Skript `src/run_batch_predictions.py` liest eine JSONL-Datei (eine JSON-Zeile pro Game-State)
und schreibt eine CSV mit den Vorhersagen.

Beispiel-Aufruf:

```bash
python3 src/run_batch_predictions.py --input data/examples/call_states_batch_example.jsonl \
	--output models/batch_predictions_example.csv
```

Die Ausgabe-CSV enthält die Spalten: `match_id, frame_id, predicted_call, p_predicted, p_second, second_call, all_probs_json`.

Eine kleine Beispiel-JSONL befindet sich in `data/examples/call_states_batch_example.jsonl`.

Feature Schema for real matches
-------------------------------

When converting real match exports into the pipeline format, each JSON line
must follow the canonical Game-State schema. This is what the training and
inference scripts expect.

Required keys in each JSONL line (types summarized):
- `match_id` (string)
- `frame_id` (int or string)
- `zone_phase` (string)
- `zone_index` (int)
- `alive_players` (int)
- `teammates_alive` (int)
- `storm_edge_dist` (int/float)
- `mats_total` (int)
- `surge_above` (int/float)
- `height_status` (string)
- `position_type` (string)
- `outcome_placement` (int, optional at labeling time)
- `outcome_alive_time` (int, optional at labeling time)

Use the converter skeleton `src/osirion_to_call_states_jsonl.py` to convert
Osirion/API/Replay exports to this canonical JSONL format. The script writes
`call_states_YYYYMMDD.jsonl` to the chosen output directory and is compatible
with `src/run_batch_predictions.py` and the training script.

Labeling helper
---------------

I included a small helper `src/label_helper.py` to prepare and finalize manual
labels. Workflow:

1. Prepare a labeling CSV from your JSONL / CSV:

```bash
python3 src/label_helper.py prepare --input data/examples/call_states_batch_example.jsonl
```

This writes `data/call_states_to_label.csv` (adds `row_id` and empty `best_call`).

2. Manually fill the `best_call` column in the CSV (use one of the allowed
	labels: `stick_deadside, play_frontside, take_height, stabilize_box, look_for_refresh, drop_low`).

3. Finalize the labeled file (validates labels and writes final training CSV):

```bash
python3 src/label_helper.py finalize --input data/call_states_labeled.csv
```

This writes `data/call_states_real_labeled.csv` which is ready for training.

4. Train on your real labeled data:

```bash
python3 src/train_best_call_baseline.py --data data/call_states_real_labeled.csv
```

The `label_helper.py` script ensures labels are valid and prevents accidental
training on incomplete label sets.

Available Models
----------------

The training script supports two classifier types:

- `rf` (RandomForest, default): Fast baseline, compatible with ONNX export.
- `hgb` (HistGradientBoosting): Often more accurate but slower; no ONNX export yet.

To train a specific model:

```bash
python3 src/train_best_call_baseline.py --data data/processed/call_states_synth_large.csv --model rf
python3 src/train_best_call_baseline.py --data data/processed/call_states_synth_large.csv --model hgb
```

To compare both models using the same CV splits:

```bash
python3 src/train_best_call_baseline.py --data data/processed/call_states_synth_large.csv --compare
```

The comparison writes:
- `models/cv_model_comparison.csv` — side-by-side accuracy and macro F1 for both models
- Appends summary to `models/report.txt` under "Model Comparison" section

Multimodal Model (PyTorch: CNN + Tabular Fusion)
-------------------------------------------------

For enhanced predictions using both screenshots and tabular game-state features,
use the multimodal model. This combines:

- **Vision branch**: Small CNN (3 Conv layers) on 224×224 screenshots → visual features
- **Tabular branch**: Dense network on 8 numeric game-state features → tabular features
- **Fusion**: Concatenated features → 6-class classifier

### Training with Cross-Validation

```bash
python3 src/multimodal_model.py \
  --data data/call_states_real_labeled.csv \
  --images data/dummy_screenshots \
  --epochs 20 \
  --n-splits 5 \
  --batch-size 8
```

### Advanced Features

**Early Stopping**: Stop training if validation F1 does not improve:

```bash
python3 src/multimodal_model.py \
  --data data/call_states_real_labeled.csv \
  --images data/dummy_screenshots \
  --epochs 50 \
  --patience 3 \
  --eval-interval 2
```

- `--patience 3`: Stop if no F1 improvement for 3 eval intervals (default: 3)
- `--eval-interval 2`: Evaluate every 2 epochs (default: 1)

**TensorBoard Monitoring**: Real-time loss, accuracy, and F1 curves:

```bash
python3 src/multimodal_model.py \
  --data data/call_states_real_labeled.csv \
  --images data/dummy_screenshots \
  --epochs 20 \
  --tensorboard \
  --verbose

# In another terminal:
tensorboard --logdir runs/
```

**Final Model Training**: Train on full dataset after CV (for production deployment):

```bash
python3 src/multimodal_model.py \
  --data data/call_states_real_labeled.csv \
  --images data/dummy_screenshots \
  --epochs 20 \
  --save-final
```

This saves the final model to `models/multimodal_final.pth`.

### Model Outputs

After training, the script creates:

- `models/multimodal_cv_results.csv` — per-fold accuracy and macro F1
- `models/checkpoints/fold_{N}/best_model.pth` — best checkpoint for each fold (selected by validation F1)
- `models/multimodal_final.pth` — final model trained on full dataset (if `--save-final`)
- `runs/multimodal_YYYYMMDD_HHMMSS/` — TensorBoard event files (if `--tensorboard`)

### Data Requirements

The multimodal model expects:

1. **Labeled CSV** with columns: `match_id, frame_id, best_call, ...` + tabular features
2. **Screenshot directory** with optional `metadata.jsonl` mapping `(match_id, frame_id)` → image paths
   - If metadata missing, model uses zero tensors (learns from tabular features only)

Example directory structure:

```
data/dummy_screenshots/
├── metadata.jsonl
├── match_001_frame_0001.jpg
└── match_001_frame_0002.jpg
```

Metadata JSONL format (one line per image):

```json
{"match_id": "match_001", "frame_id": "0001", "image_path": "data/dummy_screenshots/match_001_frame_0001.jpg"}
```

### Notes

- Model runs on CPU by default (CUDA auto-detected if available)
- Missing images are treated as zero tensors (tabular features still contribute)
- Early stopping and checkpointing ensure best generalization
- TensorBoard provides detailed training curves for debugging

REST API & Production Deployment
---------------------------------

A FastAPI-based REST server enables production inference. The API provides:

- **Single predictions**: `POST /predict` — one game state at a time
- **Batch predictions**: `POST /batch-predict` — multiple game states (up to 1000)
- **Health check**: `GET /health` — service status and available models
- **Model info**: `GET /models` — detailed model metadata

### Starting the API Server

**Standalone (development):**

```bash
python3 src/api_server.py --port 8000 --reload
```

Then test:

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "zone_phase": "mid",
    "zone_index": 5,
    "alive_players": 30,
    "teammates_alive": 3,
    "storm_edge_dist": 150.5,
    "mats_total": 400,
    "surge_above": 10,
    "height_status": "mid",
    "position_type": "edge",
    "match_id": "match_001",
    "frame_id": "frame_001"
  }'

# Batch predictions
curl -X POST http://localhost:8000/batch-predict \
  -H "Content-Type: application/json" \
  -d '{
    "examples": [
      {"zone_phase": "mid", "zone_index": 5, ...},
      {"zone_phase": "late", "zone_index": 7, ...}
    ],
    "model": "baseline"
  }'
```

**Docker (production):**

```bash
# Build image
docker build -t predictor:latest .

# Run container
docker run -p 8000:8000 predictor:latest

# Or use docker-compose for development with hot-reload:
docker-compose up
```

### API Features

- **Request validation**: Pydantic models enforce correct schema
- **Model caching**: Pipelines and ONNX sessions loaded once per server
- **Error handling**: Detailed error messages for debugging
- **Async**: Built on FastAPI for high concurrency
- **Documentation**: Auto-generated OpenAPI docs at `http://localhost:8000/docs`

Data Quality & Validation
-------------------------

Before training or deploying on real data, validate feature consistency:

```bash
python3 src/data_validation.py \
  --input data/call_states_real_labeled.csv \
  --reference data/processed/call_states_synth_large.csv \
  --output models/validation_report.json
```

This generates a comprehensive report checking:

- **Missing values**: Percentage per column
- **Class distribution**: Valid labels + distribution shift detection (Chi-square test)
- **Numeric features**: Range checks, outlier detection, distribution tests (KS test)
- **Categorical features**: Valid values, frequency counts
- **Duplicates**: By row and by (match_id, frame_id) key

Output is a JSON report with a summary status: `PASS`, `WARNING`, or `FAIL`.

Example report excerpt:

```json
{
  "summary": {
    "total_issues": 0,
    "validation_status": "PASS"
  },
  "checks": {
    "missing_values": {},
    "class_distribution": {
      "distribution": {"stick_deadside": 0.30, ...},
      "invalid_classes": []
    },
    "numeric_features": {
      "alive_players": {
        "mean": 50.1,
        "min": 20,
        "max": 80,
        "out_of_bounds": {"count": 0, "percentage": 0.0}
      }
    }
  }
}
```

Tier 2: Model Improvement
--------------------------

Advanced techniques to improve model performance:

### 1. Hyperparameter Tuning with Optuna

Automatically search for optimal hyperparameters using Bayesian optimization:

```bash
python3 src/tune_multimodal.py \
  --data data/call_states_real_labeled.csv \
  --images data/dummy_screenshots \
  --n-trials 20 \
  --cv-folds 3 \
  --epochs 10
```

This tunes:
- Learning rate (log-uniform: 1e-4 to 1e-2)
- Batch size (categorical: 4, 8, 16, 32)
- Conv filters (categorical: 16, 32, 64)
- Dense hidden units (categorical: 32, 64, 128)
- Dropout rate (continuous: 0.0 to 0.5)
- Early stopping patience (int: 2 to 5)
- Eval interval (int: 1 to 3)

Output:
- `models/tuning_results_YYYYMMDD_HHMMSS.json` — best params, trial history, all metrics
- Console output with recommended hyperparameters for next training run

### 2. Feature Engineering

Enhance baseline models with engineered features:

```bash
python3 src/train_best_call_baseline.py \
  --data data/call_states_real_labeled.csv \
  --feature-engineering \
  --importance
```

Automatically creates:
- Ratio features: health_ratio, storm_threat, mats_per_player
- Interaction features: zone_storm_interaction, players_mats_interaction
- Log transforms: log_mats, log_outcome_placement, log_outcome_alive_time
- Z-score normalization via StandardScaler

Output:
- `models/feature_importance.json` — ranked permutation feature importance
- Improved baseline model (typically +5-15% F1)

### 3. Larger Dataset Generation

Generate more training data:

```bash
python3 src/generate_large_dataset.py --n 5000 --seed 42
python3 src/train_best_call_baseline.py \
  --data data/processed/call_states_large.csv \
  --feature-engineering
```

Features:
- 5000+ samples with realistic feature correlations
- Balanced class distribution
- Output: `data/processed/call_states_large.csv`

### 4. End-to-End Benchmarking

Compare models before/after improvements:

```bash
python3 src/benchmark_tier2.py --quick   # 2-fold, 5 epochs
python3 src/benchmark_tier2.py --full    # 5-fold, 20 epochs
```

Benchmarks:
1. Baseline RF (900 samples, no FE)
2. Baseline RF + FE (900 samples)
3. Baseline RF + FE (5000+ samples)
4. Multimodal PyTorch (900 samples)
5. Multimodal PyTorch (5000+ samples)

Output: `models/benchmark_results_YYYYMMDD_HHMMSS.json` with detailed metrics

### Recommended Tier 2 Workflow

1. **Feature Engineering** (fast, interpretable):
   ```bash
   python3 src/train_best_call_baseline.py \
     --data data/call_states_real_labeled.csv \
     --feature-engineering --importance
   ```
   Expected gain: +5-15% F1

2. **Larger Dataset** (if more data available):
   ```bash
   python3 src/generate_large_dataset.py --n 5000
   python3 src/train_best_call_baseline.py \
     --data data/processed/call_states_large.csv \
     --feature-engineering
   ```
   Expected gain: +20-30% F1

3. **Hyperparameter Tuning** (for multimodal):
   ```bash
   python3 src/tune_multimodal.py --n-trials 30 --epochs 20
   ```
   Expected gain: +10-20% F1

4. **Benchmark** final system:
   ```bash
   python3 src/benchmark_tier2.py --full
   ```
