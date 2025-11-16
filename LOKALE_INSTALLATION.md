# 💻 Lokale Installation auf deinem PC

Komplette Anleitung, um das Predictor-Modell außerhalb von GitHub auf deinem PC zu nutzen.

---

## 📋 Voraussetzungen

- **Python 3.11 oder 3.12** (empfohlen)
- **pip** (Python Package Manager)
- **10 MB freier Speicherplatz** (für Modell + Code)

---

## 🚀 Schnell-Installation (3 Schritte)

### Schritt 1: Repository herunterladen

**Option A: Mit Git (empfohlen)**
```bash
git clone https://github.com/sauremilk/Predictor.git
cd Predictor
```

**Option B: Als ZIP herunterladen**
1. Gehe zu: https://github.com/sauremilk/Predictor
2. Klicke auf **Code** → **Download ZIP**
3. Entpacke die ZIP-Datei
4. Öffne Terminal/CMD im entpackten Ordner

### Schritt 2: Python-Dependencies installieren

```bash
pip install -r Vorhersage-Modell/requirements.txt
```

**Falls pip nicht funktioniert, versuche:**
```bash
# Windows
python -m pip install -r Vorhersage-Modell/requirements.txt

# macOS/Linux
python3 -m pip install -r Vorhersage-Modell/requirements.txt
```

### Schritt 3: Test ausführen

```bash
# Windows
python quickstart.py

# macOS/Linux
python3 quickstart.py
```

**Erwartete Ausgabe:**
```
📦 Lade Modell von: ...
✅ Modell geladen
🎮 Situation: mid game, 25 players alive
📊 AI Empfehlung: take_height
🎯 Confidence: 77%
```

✅ **Fertig! Das Modell läuft jetzt auf deinem PC!**

---

## 📦 Minimale Installation (nur nötige Dateien)

Wenn du nicht das ganze Repository brauchst, reichen diese Dateien:

### Benötigte Dateien:
```
Dein-Ordner/
├── predictor_lib.py                    # Bibliothek (4 KB)
├── quickstart.py                        # Beispiel (2 KB)
└── Vorhersage-Modell/
    └── models/
        └── baseline_pipeline_final.joblib   # Modell (122 KB)
```

### Installation:

**1. Ordnerstruktur erstellen:**
```bash
mkdir -p MeinPredictor/Vorhersage-Modell/models
cd MeinPredictor
```

**2. Dateien kopieren:**

Lade diese 3 Dateien aus dem GitHub-Repo:
- `predictor_lib.py` → in `MeinPredictor/`
- `quickstart.py` → in `MeinPredictor/`
- `Vorhersage-Modell/models/baseline_pipeline_final.joblib` → in `MeinPredictor/Vorhersage-Modell/models/`

**3. Dependencies installieren:**
```bash
pip install pandas scikit-learn joblib numpy
```

**4. Testen:**
```bash
python quickstart.py
```

---

## 🐍 Python-Version prüfen

```bash
# Windows
python --version

# macOS/Linux
python3 --version
```

**Sollte anzeigen:** `Python 3.11.x` oder `Python 3.12.x`

**Falls Python fehlt:**
- Windows: https://www.python.org/downloads/
- macOS: `brew install python@3.12`
- Linux: `sudo apt install python3.12`

---

## 🔧 Detaillierte Installationsschritte

### Windows

```powershell
# 1. Ordner erstellen
mkdir C:\MeinPredictor
cd C:\MeinPredictor

# 2. Repository klonen (oder ZIP entpacken)
git clone https://github.com/sauremilk/Predictor.git .

# 3. Dependencies installieren
python -m pip install --upgrade pip
python -m pip install -r Vorhersage-Modell\requirements.txt

# 4. Testen
python quickstart.py
```

### macOS

```bash
# 1. Ordner erstellen
mkdir ~/MeinPredictor
cd ~/MeinPredictor

# 2. Repository klonen
git clone https://github.com/sauremilk/Predictor.git .

# 3. Dependencies installieren
pip3 install --upgrade pip
pip3 install -r Vorhersage-Modell/requirements.txt

# 4. Testen
python3 quickstart.py
```

### Linux (Ubuntu/Debian)

```bash
# 1. Python sicherstellen
sudo apt update
sudo apt install python3.12 python3-pip

# 2. Ordner erstellen
mkdir ~/MeinPredictor
cd ~/MeinPredictor

# 3. Repository klonen
git clone https://github.com/sauremilk/Predictor.git .

# 4. Virtual Environment (optional, aber empfohlen)
python3 -m venv venv
source venv/bin/activate

# 5. Dependencies installieren
pip install -r Vorhersage-Modell/requirements.txt

# 6. Testen
python3 quickstart.py
```

---

## 🎯 Nach der Installation

### Grundlegende Nutzung:

```python
# Erstelle test.py
from predictor_lib import PredictorModel

predictor = PredictorModel()
result = predictor.predict(
    zone_phase="mid",
    alive_players=30,
    teammates_alive=3,
    height_status="high",
    position_type="edge"
)

print(f"Empfehlung: {result['predicted_call']}")
print(f"Confidence: {result['confidence']:.0%}")
```

```bash
# Ausführen
python test.py
```

### In eigenem Projekt nutzen:

**Option 1: Direkt kopieren**
```bash
# Kopiere predictor_lib.py in dein Projekt
cp predictor_lib.py /pfad/zu/deinem/projekt/

# Kopiere Modell
cp -r Vorhersage-Modell /pfad/zu/deinem/projekt/
```

**Option 2: Als Package installieren**
```python
# In deinem Projekt:
import sys
sys.path.append('/pfad/zu/MeinPredictor')

from predictor_lib import PredictorModel
```

---

## 📚 Verfügbare Tools

Nach der Installation stehen dir zur Verfügung:

### 1. Python-Bibliothek (empfohlen)
```bash
python quickstart.py           # Einfaches Beispiel
python predictor_lib.py        # Alle Features
python direct_prediction.py    # Standalone-Demo
```

### 2. Batch-Processing
```bash
python batch_predict.py        # CSV-Verarbeitung
```

### 3. CLI-Tool
```bash
python predict_cli.py --zone mid --players 20 --team 3 --height high --position edge
```

### 4. REST API (optional)
```bash
cd Vorhersage-Modell
python -m uvicorn src.api_server:app --host 0.0.0.0 --port 8000

# Oder mit Management-Skript:
./manage_api.sh start
```

---

## 🐍 Virtual Environment (empfohlen)

Verhindert Konflikte mit anderen Python-Projekten:

```bash
# Erstellen
python -m venv predictor_env

# Aktivieren
# Windows:
predictor_env\Scripts\activate
# macOS/Linux:
source predictor_env/bin/activate

# Dependencies installieren
pip install -r Vorhersage-Modell/requirements.txt

# Nutzen
python quickstart.py

# Deaktivieren (wenn fertig)
deactivate
```

---

## 🔍 Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'sklearn'"

**Lösung:**
```bash
pip install scikit-learn
# oder
pip install -r Vorhersage-Modell/requirements.txt
```

### Problem: "FileNotFoundError: Modell nicht gefunden"

**Lösung 1:** Prüfe Pfad
```bash
ls -la Vorhersage-Modell/models/baseline_pipeline_final.joblib
```

**Lösung 2:** Expliziter Pfad im Code
```python
predictor = PredictorModel(
    model_path="Vorhersage-Modell/models/baseline_pipeline_final.joblib"
)
```

### Problem: "python: command not found"

**Lösung:**
```bash
# Versuche python3 statt python
python3 quickstart.py

# Oder installiere Python:
# Windows: https://www.python.org/downloads/
# macOS: brew install python
# Linux: sudo apt install python3
```

### Problem: "Permission denied"

**Lösung (macOS/Linux):**
```bash
chmod +x predict_cli.py
chmod +x quickstart.py
```

### Problem: Alte Python-Version (< 3.11)

**Option 1:** Python updaten
```bash
# Windows: Neue Version von python.org installieren
# macOS: brew upgrade python
# Linux: sudo apt install python3.12
```

**Option 2:** Mit alter Version (3.8+) nutzen
```bash
# Funktioniert auch mit Python 3.8-3.10, aber 3.11+ empfohlen
pip install --upgrade scikit-learn pandas numpy
```

---

## 📊 Minimale System-Anforderungen

- **CPU:** Beliebig (auch alte CPUs funktionieren)
- **RAM:** 512 MB frei
- **Speicher:** 50 MB
- **OS:** Windows 10+, macOS 10.14+, Ubuntu 20.04+
- **Python:** 3.8+ (3.11+ empfohlen)

**Performance:**
- Single Prediction: < 1ms
- 1000 Predictions: ~50ms
- Model Loading: ~200ms (einmalig beim Start)

---

## 🚀 Produktive Nutzung

### In deine Anwendung integrieren:

```python
# main.py
from predictor_lib import PredictorModel

class GameAI:
    def __init__(self):
        # Modell einmal beim Start laden
        self.predictor = PredictorModel()
        print("✅ AI geladen und bereit!")
    
    def get_recommendation(self, game_state):
        """Hole AI-Empfehlung für aktuelle Situation"""
        result = self.predictor.predict(
            zone_phase=game_state['zone'],
            alive_players=game_state['players'],
            teammates_alive=game_state['team'],
            height_status=game_state['height'],
            position_type=game_state['position']
        )
        
        return {
            'call': result['predicted_call'],
            'confidence': result['confidence'],
            'alternatives': self.predictor.get_top_n_predictions(
                **game_state, n=3
            )
        }

# Nutzung
ai = GameAI()

# In deiner Game-Loop:
current_state = {
    'zone': 'mid',
    'players': 25,
    'team': 3,
    'height': 'high',
    'position': 'edge'
}

recommendation = ai.get_recommendation(current_state)
print(f"AI empfiehlt: {recommendation['call']} ({recommendation['confidence']:.0%})")
```

---

## 📦 Distribution

Wenn du deine App mit dem Modell verteilen willst:

### Dateien einpacken:
```
DeinApp/
├── dein_code.py
├── predictor_lib.py
└── models/
    └── baseline_pipeline_final.joblib
```

### requirements.txt für User:
```
pandas>=2.0.0
scikit-learn>=1.3.0
joblib>=1.3.0
numpy>=1.24.0
```

### Installation für End-User:
```bash
pip install -r requirements.txt
python dein_code.py
```

---

## 🔄 Updates

Modell aktualisieren:

```bash
cd Predictor
git pull origin main
```

Oder manuell:
1. Lade neue `baseline_pipeline_final.joblib` von GitHub
2. Ersetze alte Datei in `Vorhersage-Modell/models/`

---

## 💡 Next Steps

Nach erfolgreicher Installation:

1. **Teste Quickstart:** `python quickstart.py`
2. **Lies Dokumentation:** `PYTHON_INTEGRATION.md`
3. **Probiere Beispiele:** `predictor_lib.py`
4. **Integriere in dein Projekt:** Siehe Abschnitt "Produktive Nutzung"

---

## 📞 Support

**Dokumentation:**
- `PYTHON_INTEGRATION.md` - API-Referenz
- `ROBUSTE_NUTZUNG.md` - Alle Nutzungsmethoden
- `README.md` - Projekt-Übersicht

**Beispiele:**
- `quickstart.py` - Einfacher Start
- `predictor_lib.py` - Alle Features
- `batch_predict.py` - CSV-Batch-Processing
- `predict_cli.py` - Command-Line Interface

**Bei Problemen:**
1. Prüfe Python-Version: `python --version`
2. Prüfe Installation: `pip list | grep scikit`
3. Teste Modell: `python quickstart.py`
4. Siehe Troubleshooting-Abschnitt oben

---

## ✅ Installations-Checkliste

- [ ] Python 3.11+ installiert
- [ ] Repository heruntergeladen (Git oder ZIP)
- [ ] Dependencies installiert (`pip install -r ...`)
- [ ] Test erfolgreich (`python quickstart.py`)
- [ ] Dokumentation gelesen (`PYTHON_INTEGRATION.md`)
- [ ] Eigenen Test-Code geschrieben
- [ ] In eigenes Projekt integriert

**Alles ✅? Dann bist du fertig! 🎉**

