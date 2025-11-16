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