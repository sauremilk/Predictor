# 📦 Benötigte Dateien für lokale Installation

## ✅ MINIMAL-INSTALLATION (nur 2 Dateien nötig!)

Diese Dateien MUSST du herunterladen:

### 1. Python-Bibliothek
```
📄 predictor_lib.py (13 KB)
📍 Speicherort: /workspaces/Predictor/predictor_lib.py
🔗 Download: https://raw.githubusercontent.com/sauremilk/Predictor/main/predictor_lib.py
```

### 2. Modell-Datei  
```
📄 baseline_pipeline_final.joblib (122 KB)
📍 Speicherort: /workspaces/Predictor/Vorhersage-Modell/models/baseline_pipeline_final.joblib
🔗 Download: https://github.com/sauremilk/Predictor/raw/main/Vorhersage-Modell/models/baseline_pipeline_final.joblib
```

**Das wars! Mit diesen 2 Dateien funktioniert alles.**

---

## 📂 Ordnerstruktur erstellen

Erstelle auf deinem PC:

```
MeinPredictor/
├── predictor_lib.py                          ← Datei 1
└── Vorhersage-Modell/
    └── models/
        └── baseline_pipeline_final.joblib    ← Datei 2
```

**Windows (CMD):**
```cmd
mkdir MeinPredictor
cd MeinPredictor
mkdir Vorhersage-Modell\models
```

**macOS/Linux (Terminal):**
```bash
mkdir -p MeinPredictor/Vorhersage-Modell/models
cd MeinPredictor
```

---

## 🔧 Installation

### 1. Python-Packages installieren

```bash
# Windows
python -m pip install pandas scikit-learn joblib numpy

# macOS/Linux
pip3 install pandas scikit-learn joblib numpy
```

### 2. Test (erstelle test.py)

```python
# test.py
from predictor_lib import PredictorModel

predictor = PredictorModel()
result = predictor.predict(
    zone_phase="mid",
    alive_players=25,
    teammates_alive=3,
    height_status="high",
    position_type="edge"
)

print(f"Empfehlung: {result['predicted_call']}")
print(f"Confidence: {result['confidence']:.0%}")
```

### 3. Ausführen

```bash
# Windows
python test.py

# macOS/Linux
python3 test.py
```

**Erwartete Ausgabe:**
```
📦 Lade Modell von: ...
✅ Modell geladen
Empfehlung: take_height
Confidence: 77%
```

✅ **Fertig! Nur 2 Dateien + pip install!**

---

## 🎯 OPTIONAL: Zusätzliche nützliche Dateien

Wenn du mehr Features willst:

### Quickstart-Beispiel (zum Testen)
```
📄 quickstart.py (1.4 KB)
🔗 https://raw.githubusercontent.com/sauremilk/Predictor/main/quickstart.py
```

### CLI-Tool (Command-Line)
```
📄 predict_cli.py (6 KB)
🔗 https://raw.githubusercontent.com/sauremilk/Predictor/main/predict_cli.py
```

### Batch-Processing (CSV-Verarbeitung)
```
📄 batch_predict.py (2 KB)
🔗 https://raw.githubusercontent.com/sauremilk/Predictor/main/batch_predict.py
```

---

## 📥 Download-Methoden

### Option 1: Manueller Download (Browser)

1. **predictor_lib.py:**
   - Öffne: https://raw.githubusercontent.com/sauremilk/Predictor/main/predictor_lib.py
   - Rechtsklick → Speichern unter → `predictor_lib.py`

2. **baseline_pipeline_final.joblib:**
   - Öffne: https://github.com/sauremilk/Predictor/blob/main/Vorhersage-Modell/models/baseline_pipeline_final.joblib
   - Klicke "Download" Button → Speichern

### Option 2: Mit wget (Linux/macOS)

```bash
# In MeinPredictor/ Ordner
wget https://raw.githubusercontent.com/sauremilk/Predictor/main/predictor_lib.py

# Modell
mkdir -p Vorhersage-Modell/models
cd Vorhersage-Modell/models
wget https://github.com/sauremilk/Predictor/raw/main/Vorhersage-Modell/models/baseline_pipeline_final.joblib
cd ../..
```

### Option 3: Mit curl (macOS/Linux/Windows PowerShell)

```bash
# Python-Bibliothek
curl -O https://raw.githubusercontent.com/sauremilk/Predictor/main/predictor_lib.py

# Modell
mkdir -p Vorhersage-Modell/models
curl -L -o Vorhersage-Modell/models/baseline_pipeline_final.joblib \
  https://github.com/sauremilk/Predictor/raw/main/Vorhersage-Modell/models/baseline_pipeline_final.joblib
```

### Option 4: Mit PowerShell (Windows)

```powershell
# Python-Bibliothek
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/sauremilk/Predictor/main/predictor_lib.py" -OutFile "predictor_lib.py"

# Ordner erstellen
New-Item -ItemType Directory -Force -Path "Vorhersage-Modell\models"

# Modell
Invoke-WebRequest -Uri "https://github.com/sauremilk/Predictor/raw/main/Vorhersage-Modell/models/baseline_pipeline_final.joblib" -OutFile "Vorhersage-Modell\models\baseline_pipeline_final.joblib"
```

---

## ✅ Verifizierung

Prüfe ob alles da ist:

```bash
# Windows
dir predictor_lib.py
dir Vorhersage-Modell\models\baseline_pipeline_final.joblib

# macOS/Linux
ls -lh predictor_lib.py
ls -lh Vorhersage-Modell/models/baseline_pipeline_final.joblib
```

**Sollte zeigen:**
- `predictor_lib.py` → ~13 KB
- `baseline_pipeline_final.joblib` → ~122 KB

---

## 🚀 Komplettes Repository (Alternative)

Wenn du ALLES willst (alle Tools, Docs, Beispiele):

```bash
git clone https://github.com/sauremilk/Predictor.git
cd Predictor
pip install pandas scikit-learn joblib numpy
python quickstart.py
```

**Oder als ZIP:**
https://github.com/sauremilk/Predictor/archive/refs/heads/main.zip

---

## 📊 Dateigrößen-Übersicht

| Datei | Größe | Erforderlich |
|-------|-------|--------------|
| predictor_lib.py | 13 KB | ✅ JA |
| baseline_pipeline_final.joblib | 122 KB | ✅ JA |
| quickstart.py | 1.4 KB | Optional |
| predict_cli.py | 6 KB | Optional |
| batch_predict.py | 2 KB | Optional |
| Dokumentation (.md) | ~50 KB | Optional |

**Total minimal: 135 KB (nur 2 Dateien!)**

---

## 🔍 Troubleshooting

### "Modell nicht gefunden"
→ Prüfe Ordnerstruktur, muss exakt so sein:
```
dein-ordner/
├── predictor_lib.py
└── Vorhersage-Modell/
    └── models/
        └── baseline_pipeline_final.joblib
```

### "ModuleNotFoundError"
→ Installiere Packages:
```bash
pip install pandas scikit-learn joblib numpy
```

### "Permission denied" (Linux/macOS)
→ Keine Sorge, nur Python-Dateien brauchen keine Ausführungsrechte

---

## 💡 Quick Start Zusammenfassung

```bash
# 1. Ordner erstellen
mkdir -p MeinPredictor/Vorhersage-Modell/models
cd MeinPredictor

# 2. Dateien herunterladen (siehe Download-Methoden oben)

# 3. Python-Packages
pip install pandas scikit-learn joblib numpy

# 4. Testen (erstelle test.py wie oben gezeigt)
python test.py
```

**Nur 2 Dateien (135 KB) + pip install = Fertig!** 🎉

---

## 📚 Weiterführende Links

- **API-Dokumentation:** PYTHON_INTEGRATION.md (im Repo)
- **Beispiele:** quickstart.py, batch_predict.py (im Repo)
- **Repository:** https://github.com/sauremilk/Predictor

---

**Kurz gesagt: Du brauchst nur `predictor_lib.py` + `baseline_pipeline_final.joblib` + `pip install` – das wars!**
