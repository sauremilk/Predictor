# 🚀 Direkte Python-Integration - Quickstart

Die robusteste Methode zur Nutzung des Predictor-Modells.

## ⚡ Super Quick Start (5 Zeilen)

```python
from predictor_lib import PredictorModel

predictor = PredictorModel()
result = predictor.predict(
    zone_phase="mid", alive_players=25, teammates_alive=3,
    height_status="high", position_type="edge"
)
print(result['predicted_call'])  # → 'take_height'
```

## 📦 Dateien

- **`predictor_lib.py`** - Hauptbibliothek (kopiere diese in dein Projekt)
- **`quickstart.py`** - Einfaches Beispiel zum Starten
- **`direct_prediction.py`** - Standalone-Demo ohne Import

## 🎯 Anwendungsfälle

### 1. Single Prediction

```python
from predictor_lib import PredictorModel

predictor = PredictorModel()

result = predictor.predict(
    zone_phase="late",
    alive_players=15,
    teammates_alive=2,
    height_status="low",
    position_type="corner",
    mats_total=150,
    storm_edge_dist=50.0
)

print(f"Call: {result['predicted_call']}")
print(f"Confidence: {result['confidence']:.0%}")
```

### 2. Top-N Empfehlungen

```python
top_3 = predictor.get_top_n_predictions(
    zone_phase="mid",
    alive_players=30,
    teammates_alive=3,
    height_status="mid",
    position_type="edge",
    n=3
)

for call in top_3:
    print(f"{call['call']}: {call['probability']:.0%}")
```

### 3. Batch Processing

```python
situations = [
    {"zone_phase": "early", "alive_players": 80, 
     "teammates_alive": 4, "height_status": "mid", "position_type": "center"},
    {"zone_phase": "late", "alive_players": 10,
     "teammates_alive": 2, "height_status": "low", "position_type": "corner"}
]

results = predictor.predict_batch(situations)

for result in results:
    print(result['predicted_call'])
```

### 4. DataFrame/CSV Integration

```python
import pandas as pd

# CSV einlesen
df = pd.read_csv("game_states.csv")

# Predictions hinzufügen
df_with_predictions = predictor.predict_from_dataframe(df)

# Als CSV speichern
df_with_predictions.to_csv("predictions.csv", index=False)
```

## 🔧 API-Referenz

### `PredictorModel(model_path=None)`

Initialisiert das Modell.

**Parameter:**
- `model_path` (optional): Pfad zur Pipeline (auto-detect wenn nicht angegeben)

### `predict(...)`

Einzelne Prediction.

**Required Parameters:**
- `zone_phase`: str - `"early"`, `"mid"`, `"late"`
- `alive_players`: int - Anzahl lebender Spieler
- `teammates_alive`: int - Anzahl Teammitglieder  
- `height_status`: str - `"low"`, `"mid"`, `"high"`
- `position_type`: str - `"edge"`, `"corner"`, `"center"`

**Optional Parameters:**
- `zone_index`: int - Zone 1-9 (default: 5)
- `storm_edge_dist`: float - Distanz zum Storm (default: 100.0)
- `mats_total`: int - Materialien (default: 300)
- `surge_above`: int - Surge-Vorteil (default: 10)
- `return_probabilities`: bool - Wahrscheinlichkeiten ausgeben (default: True)

**Returns:**
```python
{
    "predicted_call": "take_height",
    "confidence": 0.77,
    "probabilities": {
        "play_frontside": 0.16,
        "stick_deadside": 0.07,
        "take_height": 0.77
    }
}
```

### `predict_batch(game_states, return_probabilities=True)`

Batch-Predictions für Performance.

**Parameter:**
- `game_states`: List[Dict] - Liste von Game States
- `return_probabilities`: bool - Wahrscheinlichkeiten zurückgeben

**Returns:** List[Dict] - Liste von Prediction-Results

### `predict_from_dataframe(df)`

Predictions direkt auf DataFrame.

**Parameter:**
- `df`: pandas.DataFrame - DataFrame mit Features

**Returns:** pandas.DataFrame - Originaldaten + `predicted_call` + `confidence`

### `get_top_n_predictions(..., n=3)`

Top-N wahrscheinlichste Calls.

**Returns:**
```python
[
    {"call": "take_height", "probability": 0.77},
    {"call": "play_frontside", "probability": 0.16},
    {"call": "stick_deadside", "probability": 0.07}
]
```

## 🎮 Live-Tests

```bash
# Quickstart-Beispiel
python3 quickstart.py

# Alle Features testen
python3 predictor_lib.py

# Standalone ohne Import
python3 direct_prediction.py
```

## 💡 Integration-Template

```python
from predictor_lib import PredictorModel

class GameAnalyzer:
    def __init__(self):
        # Modell einmal beim Start laden
        self.predictor = PredictorModel()
    
    def analyze_situation(self, game_data):
        """Analysiere aktuelle Game-Situation"""
        result = self.predictor.predict(
            zone_phase=game_data['zone'],
            alive_players=game_data['players'],
            teammates_alive=game_data['team_size'],
            height_status=game_data['height'],
            position_type=game_data['position'],
            mats_total=game_data.get('mats', 300)
        )
        
        # Entscheidungslogik
        if result['confidence'] > 0.7:
            return f"Starke Empfehlung: {result['predicted_call']}"
        else:
            # Bei unsicheren Predictions: Top-3 anzeigen
            top_3 = self.predictor.get_top_n_predictions(
                **game_data, n=3
            )
            return f"Optionen: {[c['call'] for c in top_3]}"

# Nutzung
analyzer = GameAnalyzer()
recommendation = analyzer.analyze_situation(current_game_state)
print(recommendation)
```

## ⚡ Performance

- **Single Prediction**: < 1ms
- **Batch (100 predictions)**: ~10ms  
- **Batch (1000 predictions)**: ~50ms
- **Model Loading**: ~200ms (einmalig beim Start)

**Tipp:** Lade das Modell nur einmal beim App-Start, nicht bei jeder Prediction!

## 🔍 Troubleshooting

### "FileNotFoundError: Modell nicht gefunden"

```python
# Expliziter Pfad angeben
predictor = PredictorModel(
    model_path="/absoluter/pfad/zu/baseline_pipeline_final.joblib"
)
```

### "ValueError: columns are missing"

Stelle sicher, alle Required-Parameter sind gesetzt:
- `zone_phase`, `alive_players`, `teammates_alive`
- `height_status`, `position_type`

### Performance-Optimierung

```python
# ❌ Langsam (lädt Modell 100x)
for state in states:
    predictor = PredictorModel()  # Nicht jedes Mal neu laden!
    result = predictor.predict(**state)

# ✅ Schnell (lädt Modell 1x, nutzt Batch)
predictor = PredictorModel()  # Einmalig
results = predictor.predict_batch(states)  # Vektorisiert
```

## 📊 Vergleich: Python vs. API

| Feature | Python-Integration | REST API |
|---------|-------------------|----------|
| Latency | < 1ms | 10-50ms |
| Setup | Import + 1 Zeile | Server starten |
| Dependencies | joblib, pandas, sklearn | + FastAPI, uvicorn |
| Use Case | Python-Apps | Microservices, andere Sprachen |
| Performance (1000 calls) | ~50ms | ~10s |

**Empfehlung:** Nutze Python-Integration wenn möglich!

## 📚 Weitere Beispiele

Siehe `predictor_lib.py` Abschnitt `if __name__ == "__main__"` für:
- Alle Features im Detail
- DataFrame-Workflows
- Batch-Processing-Strategien
- Integration-Patterns

---

**Fragen? Siehe auch:**
- `ROBUSTE_NUTZUNG.md` - Übersicht aller Methoden
- `batch_predict.py` - CSV-Batch-Processing
- `predict_cli.py` - Command-Line Interface
