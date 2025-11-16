# 🎯 Installation auf deinem PC - Komplettübersicht

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  GitHub Repository                                          │
│  https://github.com/sauremilk/Predictor                    │
│                                                             │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
        ┌────────────────┐
        │  Download auf  │
        │   deinen PC    │
        └────────┬───────┘
                 │
         ┌───────┴────────┐
         │                │
    ┌────▼─────┐    ┌─────▼─────┐
    │ Option 1 │    │ Option 2  │
    │ Komplett │    │  Minimal  │
    │ (5 MB)   │    │ (130 KB)  │
    └────┬─────┘    └─────┬─────┘
         │                │
         ▼                ▼
    ┌─────────────────────────┐
    │  Setup ausführen        │
    │  ─────────────────────  │
    │  Windows: setup.bat     │
    │  Linux/Mac: setup.sh    │
    │  Manual: pip install... │
    └────────┬────────────────┘
             │
             ▼
    ┌─────────────────────────┐
    │  ✅ Installation fertig! │
    └────────┬────────────────┘
             │
        ┌────┴────┐
        │ Nutzen: │
        └────┬────┘
             │
    ┌────────┴────────────┐
    │                     │
    ▼                     ▼
┌─────────────┐    ┌──────────────┐
│ Python-Code │    │   CLI-Tool   │
│             │    │              │
│ predictor = │    │ ./predict_   │
│  Model()    │    │  cli.py ...  │
│ result =    │    │              │
│  predict()  │    │              │
└─────────────┘    └──────────────┘
```

---

## 📥 **Schritt 1: Download**

### Option A: Komplett (empfohlen)
```
🔗 https://github.com/sauremilk/Predictor/archive/refs/heads/main.zip
📦 Größe: ~5 MB
📂 Enthält: Alles (Tools, Docs, Beispiele)
```

### Option B: Minimal (nur Bibliothek)
```
📄 predictor_lib.py (8 KB)
🔗 https://raw.githubusercontent.com/.../predictor_lib.py

📄 baseline_pipeline_final.joblib (122 KB)
🔗 https://github.com/.../baseline_pipeline_final.joblib
```

---

## 🔧 **Schritt 2: Installation**

### Windows
```cmd
1. ZIP entpacken
2. Doppelklick setup.bat
   ODER
   cd Predictor
   python -m pip install pandas scikit-learn joblib numpy
```

### macOS/Linux
```bash
1. ZIP entpacken / git clone
2. ./setup.sh
   ODER
   cd Predictor
   pip3 install pandas scikit-learn joblib numpy
```

---

## ✅ **Schritt 3: Testen**

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
📊 AI Empfehlung: take_height
🎯 Confidence: 77%
```

---

## 💻 **Schritt 4: Nutzen**

### In deinem Python-Code:

```python
from predictor_lib import PredictorModel

# Einmal initialisieren
predictor = PredictorModel()

# Nutzen
result = predictor.predict(
    zone_phase="mid",
    alive_players=25,
    teammates_alive=3,
    height_status="high",
    position_type="edge"
)

print(result['predicted_call'])  # → "take_height"
print(result['confidence'])      # → 0.77
```

### Als CLI-Tool:

```bash
python predict_cli.py \
  --zone mid \
  --players 25 \
  --team 3 \
  --height high \
  --position edge
```

---

## 📚 **Wichtige Dateien**

| Datei | Zweck | Größe |
|-------|-------|-------|
| `predictor_lib.py` | Haupt-Bibliothek | 8 KB |
| `baseline_pipeline_final.joblib` | Modell | 122 KB |
| `quickstart.py` | Schnelltest | 1 KB |
| `predict_cli.py` | Command-Line Tool | 7 KB |
| `batch_predict.py` | CSV-Batch-Processing | 3 KB |
| `setup.bat` / `setup.sh` | Auto-Setup | 2-3 KB |

**Total minimal:** ~130 KB (nur Lib + Modell)  
**Total komplett:** ~5 MB (alles)

---

## 🎯 **Was passt für dich?**

### Du willst es einfach nur ausprobieren:
→ **Download komplettes ZIP + setup.bat/sh**

### Du willst es in dein Projekt integrieren:
→ **Minimal-Installation (nur predictor_lib.py + Modell)**

### Du brauchst alle Features (API, Notebooks, etc.):
→ **Git clone oder komplettes ZIP**

---

## 📖 **Dokumentation**

Nach der Installation lies:

1. **SCHNELLSTART.md** ← Start hier!
2. **PYTHON_INTEGRATION.md** ← API-Referenz
3. **LOKALE_INSTALLATION.md** ← Detaillierte Anleitung
4. **ROBUSTE_NUTZUNG.md** ← Alle Nutzungsmethoden

---

## ⚡ **Quick Commands**

```bash
# Git Clone (einfachste Methode)
git clone https://github.com/sauremilk/Predictor.git
cd Predictor
./setup.sh  # oder setup.bat auf Windows

# Manuell (ohne Git)
# 1. Download ZIP von GitHub
# 2. Entpacken
# 3. cd Predictor
# 4. pip install pandas scikit-learn joblib numpy
# 5. python quickstart.py
```

---

## 🔍 **Troubleshooting**

| Problem | Lösung |
|---------|--------|
| Python nicht gefunden | Installiere von python.org (Windows: "Add to PATH"!) |
| ModuleNotFoundError | `pip install pandas scikit-learn joblib numpy` |
| Modell nicht gefunden | Prüfe Ordnerstruktur (siehe LOKALE_INSTALLATION.md) |
| Setup-Skript fehlt | Download komplett oder nutze manuelle Installation |

---

## 🎓 **Learning Path**

```
1. Download & Installation     (5 Minuten)
   ↓
2. Quickstart ausführen        (1 Minute)
   ↓
3. Eigenes Beispiel schreiben  (5 Minuten)
   ↓
4. In Projekt integrieren      (30 Minuten)
   ↓
5. Advanced Features nutzen    (nach Bedarf)
```

---

## ✅ **Checkliste**

- [ ] Repository heruntergeladen (ZIP oder Git)
- [ ] Python 3.8+ installiert (check: `python --version`)
- [ ] Dependencies installiert (`pip install ...`)
- [ ] Quickstart erfolgreich getestet
- [ ] Dokumentation gelesen (PYTHON_INTEGRATION.md)
- [ ] Eigenes Test-Skript geschrieben
- [ ] In eigenes Projekt integriert

**Alles ✅? Glückwunsch, du bist ready! 🎉**

---

## 🚀 **Next Steps**

Nach erfolgreicher Installation:

1. Probiere verschiedene Game-States aus
2. Teste Batch-Processing mit CSV-Dateien
3. Integriere in deine eigene Anwendung
4. Experimentiere mit den Parametern
5. Lies API-Dokumentation für Advanced Features

---

**📧 Support:** Siehe LOKALE_INSTALLATION.md → Troubleshooting  
**🔗 Repository:** https://github.com/sauremilk/Predictor  
**📚 Docs:** Alle .md Dateien im Root-Verzeichnis
