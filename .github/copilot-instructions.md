# Copilot Instructions for `Predictor`

## Big Picture
- Project: ML-basierte Mehrklassen-Klassifikation `best_call` + REST-API.
- Kernbereiche:
  - `src/train_best_call_baseline.py`: Baseline-Training (sklearn, optional Feature Engineering, CV, Vergleich RF/HGB).
  - `src/multimodal_model.py`: Multimodales PyTorch-Modell (CNN auf Screenshots + tabulare Features, Cross-Validation, Early Stopping, TensorBoard).
  - `src/api_server.py`: FastAPI-Server für Inferenz (Endpoints `/health`, `/models`, `/predict`, `/batch-predict`).
  - `src/data_validation.py`: Datenqualitäts-Checks (Missing Values, Range-Checks, Distribution Shifts, Klassendistr.).
  - `src/feature_engineering.py`, `src/generate_large_dataset.py`, `src/benchmark_tier2.py`, `src/tune_multimodal.py`: Tier-2-Verbesserungen (Feature-Engineering, Datengenerierung, Benchmarking, Hyperparametertuning).
- Modelle & Artefakte liegen in `models/` (Joblib-Pipelines, ONNX, PyTorch-Checkpoints, Reports) und `data/` (Demo-/Synth-/Real-Daten).

## Wichtige Workflows
- Baseline-Training (kleiner Demo-Datensatz):
  - `python3 src/train_best_call_baseline.py --data data/call_states_demo.csv`.
- Training auf realen Labels:
  - Label-Workflow über `src/label_helper.py` (`prepare` → manuell labeln → `finalize`).
  - Danach: `python3 src/train_best_call_baseline.py --data data/call_states_real_labeled.csv [--feature-engineering --importance]`.
- Multimodales Training:
  - `python3 src/multimodal_model.py --data data/call_states_real_labeled.csv --images data/dummy_screenshots --epochs 20 --n-splits 5`.
- ONNX-Export + Inferenz:
  - Export: `src/export_to_onnx.py` erzeugt `models/best_call_baseline.onnx` (nur Klassifizierer).
  - Single Inference: `python3 -m src.predict_best_call --input example.json` (Preprocessing via Joblib-Pipeline, Klassifikation via ONNX).
- Batch-Inferenz aus JSONL:
  - `python3 src/run_batch_predictions.py --input data/examples/call_states_batch_example.jsonl --output models/batch_predictions_example.csv`.
- Datenvalidierung vor Training/Deployment:
  - `python3 src/data_validation.py --input <candidate.csv> --reference data/processed/call_states_synth_large.csv --output models/validation_report.json`.
- Tier-2-Benchmark:
  - Schnell: `python3 src/benchmark_tier2.py --quick`; vollständig: `python3 src/benchmark_tier2.py --full` erzeugt JSON-Benchmark in `models/`.

## Architektur- und Designentscheidungen
- **Trennung Preprocessing/Modell**:
  - Baseline-Pipeline in Joblib (`models/baseline_pipeline_final.joblib`) enthält Preprocessor `pre` + Klassifizierer `clf`.
  - ONNX-Modell (`models/best_call_baseline.onnx`) kapselt nur den Klassifizierer. Preprocessing bleibt in Python (Komplexität von Imputern/OneHotEncoder).
  - Bei Inferenz immer zuerst Python-Preprocessing anwenden, dann ONNX ausführen.
- **Kanonisches Feature-Schema**:
  - Trainings-/Inferenz-Skripte erwarten konsistente Spalten (z. B. `match_id`, `frame_id`, `zone_phase`, `alive_players`, `teammates_alive`, `storm_edge_dist`, `mats_total`, `surge_above`, `height_status`, `position_type`, optional Outcome-Felder).
  - Konvertierung realer Daten erfolgt über `src/osirion_to_call_states_jsonl.py` → JSONL → CSV.
- **Strukturierte Experimente**:
  - Cross-Validation (StratifiedKFold) standardmäßig im Baseline-Training und multimodalen Modell.
  - Tuning via `src/tune_multimodal.py` mit Optuna (TPE, MedianPruner); Ergebnisse in `models/tuning_results_*.json`.
  - Feature-Engineering und Importance über `src/feature_engineering.py` und `--importance`-Flag.

## API-Server-Konventionen
- Start des Servers (Dev): `python3 src/api_server.py --port 8000 --reload`.
- Endpoints (FastAPI):
  - `GET /health`: Antwort enthält Status und verfügbare Modelle.
  - `GET /models`: Metadaten zu Modellen (Name, Typ, Pfade).
  - `POST /predict`: Single-Game-State als JSON (kanonisches Schema), liefert vorhergesagten `best_call` + Wahrscheinlichkeiten.
  - `POST /batch-predict`: Body `{ "examples": [...], "model": "baseline" }`, max. ~1000 Beispiele.
- Request-Validierung mit Pydantic; Modelle/ONNX-Sessions werden beim Start einmalig geladen und gecached.
- Multimodale Inferenz über API ist (Stand Doku) noch eingeschränkt/als zukünftiger Schritt vorgesehen – beim Hinzufügen an bestehende Modell-Lade-Mechanik anknüpfen.

## Projekt-spezifische Konventionen
- Python-Version: 3.11 (siehe Dockerfile/Umgebung im Ordner `Vorhersage-Modell/`).
- Skripte in `src/` sind als CLI-Tools konzipiert (argparse/Click-ähnlich) – neue Tools sollten sich stilistisch daran orientieren.
- Metriken: primär Accuracy, sekundär macro F1; CV-Reports werden als CSV (`models/cv_report.csv`, `models/multimodal_cv_results.csv`, `models/cv_model_comparison.csv`) abgelegt.
- Persistierte Modelle/Artefakte immer unter `models/` ablegen; generierte Datensätze unter `data/processed/`.
- Logging/Progress: Multimodales Training nutzt optional TensorBoard (`--tensorboard`), andere Skripte loggen überwiegend auf STDOUT.

## Hinweise für Änderungen
- Beim Ändern des Feature-Schemas: 
  - Konsistenz zwischen `generate_*`, `label_helper.py`, `train_best_call_baseline.py`, `multimodal_model.py`, API-Schema in `api_server.py` und Datenvalidierung sicherstellen.
  - Beispiel-Daten (`data/call_states_demo.csv`, `data/processed/*.csv`, `data/examples/*.jsonl`) mit pflegen, wenn neue Pflicht-Features eingeführt werden.
- Beim Hinzufügen neuer Modelle:
  - Training als separates Skript in `src/` mit klaren CLI-Argumenten.
  - Integration in `api_server.py` über ein neues Modell-Label und Eintrag in die Modellauswahl (z. B. Mapping von Modellnamen auf Ladefunktionen/Pfade).
- Tests/Validierung:
  - Bevor du neue Pipelines/Modelle als „stabil“ ansiehst, mindestens `data_validation.py` auf den verwendeten Datensätzen laufen lassen und CV-Metriken in `models/` persistieren.
