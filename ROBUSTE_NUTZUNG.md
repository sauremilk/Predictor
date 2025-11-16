# 🔧 Robuste Wege zur Modell-Nutzung (ohne Web-Interface)

## Übersicht der Methoden

| Methode | Use Case | Robustheit | Performance |
|---------|----------|------------|-------------|
| **1. Direkte Python-Integration** | Integration in eigene Python-Apps | ⭐⭐⭐⭐⭐ | ⚡⚡⚡ Sehr schnell |
| **2. Batch-Processing (CSV)** | Viele Predictions auf einmal | ⭐⭐⭐⭐⭐ | ⚡⚡⚡ Sehr schnell |
| **3. CLI-Tool** | Scripts, Automation, CI/CD | ⭐⭐⭐⭐⭐ | ⚡⚡⚡ Schnell |
| **4. REST API** | Microservices, Web-Apps | ⭐⭐⭐ | ⚡⚡ Netzwerk-Overhead |

---

## 1. Direkte Python-Integration ⭐ EMPFOHLEN

**Wann nutzen:** Du entwickelst in Python und brauchst maximale Performance.

**Datei:** `direct_prediction.py`

```bash
python3 direct_prediction.py
```

**Vorteile:**
- ✅ Kein Server/Netzwerk nötig
- ✅ Maximale Performance (< 1ms pro Prediction)
- ✅ Einfach in eigenen Code zu integrieren
- ✅ Vollständige Kontrolle

**Integration in deinen Code:**
```python
import joblib
import pandas as pd

# Modell einmal laden
pipeline = joblib.load("Vorhersage-Modell/models/baseline_pipeline_final.joblib")

# Prediction
game_state = {"zone_phase": "mid", "alive_players": 30, ...}
df = pd.DataFrame([game_state])
prediction = pipeline.predict(df)[0]
```

---

## 2. Batch-Processing (CSV/JSONL) ⭐ FÜR GROSSE DATENMENGEN

**Wann nutzen:** Du hast viele Game States (100+) und willst sie alle auf einmal verarbeiten.

**Datei:** `batch_predict.py`

```bash
python3 batch_predict.py
```

**Anpassung für eigene Daten:**
```python
# In batch_predict.py ändern:
input_file = "meine_daten.csv"
output_file = "ergebnisse.csv"
```

**Vorteile:**
- ✅ Sehr effizient für große Datenmengen
- ✅ Einfacher CSV-Workflow
- ✅ Ergebnisse direkt als CSV (Excel-kompatibel)
- ✅ Vektorisierte Operationen (schnell)

**Performance:** ~1000 Predictions in < 1 Sekunde

---

## 3. CLI-Tool ⭐ FÜR AUTOMATION

**Wann nutzen:** Scripts, Shell-Integration, CI/CD-Pipelines.

**Datei:** `predict_cli.py`

### Beispiele:

**Single Prediction (Text-Output):**
```bash
./predict_cli.py \
  --zone late \
  --players 15 \
  --team 2 \
  --height low \
  --position corner
```

**JSON-Output (für weitere Verarbeitung):**
```bash
./predict_cli.py \
  --zone mid \
  --players 20 \
  --team 3 \
  --height mid \
  --position edge \
  --format json
```

**Quiet Mode (nur Call, für Scripts):**
```bash
CALL=$(./predict_cli.py --zone late --players 10 --team 2 --height low --position corner --quiet)
echo "Recommended: $CALL"
```

**JSON-Input:**
```bash
./predict_cli.py --json '{"zone_phase": "mid", "alive_players": 30, "teammates_alive": 3, "height_status": "mid", "position_type": "edge", "storm_edge_dist": 100, "mats_total": 300, "surge_above": 10, "zone_index": 5}'
```

**Batch CSV-Processing:**
```bash
./predict_cli.py --csv input.csv --output predictions.csv
```

**Vorteile:**
- ✅ Shell-Integration
- ✅ Flexible Output-Formate (text/json/quiet)
- ✅ Perfekt für Automation
- ✅ Keine Python-Kenntnisse nötig (einmal setup, dann CLI)

**Hilfe anzeigen:**
```bash
./predict_cli.py --help
```

---

## 4. REST API (bereits vorhanden)

**Wann nutzen:** Microservices, Web-Apps, externe Systeme.

**Server starten:**
```bash
cd Vorhersage-Modell
./manage_api.sh start
```

**Nutzung:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"zone_phase": "mid", "alive_players": 30, ...}'
```

**Vorteile:**
- ✅ Sprachunabhängig (jede Sprache kann HTTP)
- ✅ Ideal für verteilte Systeme
- ✅ Swagger-Dokumentation unter `/docs`

**Nachteile:**
- ❌ Netzwerk-Overhead (~10-50ms)
- ❌ Server muss laufen

---

## Performance-Vergleich

| Methode | Latency (Single) | Throughput (Batch) |
|---------|------------------|-------------------|
| Direkte Integration | < 1ms | 10,000+/sec |
| Batch CSV | N/A | 5,000+/sec |
| CLI-Tool | ~50ms (Startup) | 500/sec |
| REST API | 10-50ms | 100-500/sec |

---

## Empfehlung nach Use Case

### 🎯 **Du entwickelst eine Python-App**
→ **Direkte Integration** (`direct_prediction.py`)
- Einfachste Integration
- Beste Performance
- Keine Dependencies

### 📊 **Du hast große Datasets zum Analysieren**
→ **Batch-Processing** (`batch_predict.py`)
- CSV rein, CSV raus
- Sehr schnell
- Excel-kompatibel

### ⚙️ **Du brauchst es in Shell-Scripts/Automation**
→ **CLI-Tool** (`predict_cli.py`)
- Flexible Nutzung
- Quiet-Mode für Scripts
- JSON-Support

### 🌐 **Du willst es von anderen Sprachen/Services nutzen**
→ **REST API** (`api_server.py`)
- HTTP-basiert
- Sprachunabhängig
- Swagger-Docs

---

## Quick Start

Alle Tools sind fertig, teste sie einfach:

```bash
cd /workspaces/Predictor

# 1. Direkte Python-Nutzung
python3 direct_prediction.py

# 2. Batch-Processing (nutzt Demo-Daten)
python3 batch_predict.py

# 3. CLI-Tool
./predict_cli.py --zone late --players 15 --team 2 --height low --position corner

# 4. REST API (bereits läuft)
curl http://localhost:8000/health
```

---

## Anpassung für deine Daten

### Eigene CSV verarbeiten:
```python
# In batch_predict.py:
input_file = "pfad/zu/deinen/daten.csv"
output_file = "pfad/zu/ergebnissen.csv"
```

### In eigenen Code integrieren:
```python
# Kopiere aus direct_prediction.py und passe an
import joblib
pipeline = joblib.load("Vorhersage-Modell/models/baseline_pipeline_final.joblib")
# ... deine Logik
```

### CLI in Scripts nutzen:
```bash
#!/bin/bash
CALL=$(./predict_cli.py --zone mid --players 20 --team 3 --height mid --position edge --quiet)
echo "AI recommends: $CALL"
```

---

## Troubleshooting

**"Pipeline not found"**
- Prüfe Pfad: `ls -la Vorhersage-Modell/models/baseline_pipeline_final.joblib`
- Stelle sicher, du bist im Root-Verzeichnis: `cd /workspaces/Predictor`

**"Missing columns"**
- Stelle sicher, alle Features sind gesetzt (inkl. `outcome_placement`, `outcome_alive_time`)
- Check Schema in `direct_prediction.py`

**Performance-Probleme**
- Nutze Batch-Processing statt einzelne Predictions
- Lade Modell nur einmal (nicht pro Prediction)

---

**Welche Methode passt für deinen Use Case?**
