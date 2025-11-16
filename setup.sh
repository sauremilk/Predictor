#!/bin/bash
# Automatisches Setup-Skript für lokale Installation

set -e  # Bei Fehler abbrechen

echo "╔═══════════════════════════════════════════════════════╗"
echo "║   PREDICTOR - Automatische Lokale Installation       ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo ""

# Farben für Output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Python-Command ermitteln
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    PIP_CMD="pip3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
    PIP_CMD="pip"
else
    echo -e "${RED}❌ Python nicht gefunden!${NC}"
    echo "Bitte installiere Python 3.11 oder höher:"
    echo "  - Windows: https://www.python.org/downloads/"
    echo "  - macOS: brew install python@3.12"
    echo "  - Linux: sudo apt install python3.12"
    exit 1
fi

# Python-Version prüfen
echo "🔍 Prüfe Python-Version..."
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
echo "   Gefunden: Python $PYTHON_VERSION"

# Warnung bei alter Version
MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
if [ "$MAJOR" -lt 3 ] || ([ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 8 ]); then
    echo -e "${RED}❌ Python $PYTHON_VERSION ist zu alt!${NC}"
    echo "Mindestens Python 3.8 erforderlich, 3.11+ empfohlen"
    exit 1
elif [ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 11 ]; then
    echo -e "${YELLOW}⚠️  Python $PYTHON_VERSION funktioniert, aber 3.11+ empfohlen${NC}"
else
    echo -e "${GREEN}✅ Python-Version OK${NC}"
fi

echo ""

# Dependencies installieren
echo "📦 Installiere Python-Dependencies..."
if [ -f "Vorhersage-Modell/requirements.txt" ]; then
    $PIP_CMD install -q -r Vorhersage-Modell/requirements.txt
    echo -e "${GREEN}✅ Dependencies installiert${NC}"
else
    echo -e "${YELLOW}⚠️  requirements.txt nicht gefunden, installiere Basis-Packages...${NC}"
    $PIP_CMD install -q pandas scikit-learn joblib numpy
    echo -e "${GREEN}✅ Basis-Packages installiert${NC}"
fi

echo ""

# Modell prüfen
echo "🔍 Prüfe Modell-Dateien..."
if [ -f "Vorhersage-Modell/models/baseline_pipeline_final.joblib" ]; then
    MODEL_SIZE=$(du -h "Vorhersage-Modell/models/baseline_pipeline_final.joblib" | cut -f1)
    echo "   Modell gefunden: $MODEL_SIZE"
    echo -e "${GREEN}✅ Modell vorhanden${NC}"
else
    echo -e "${RED}❌ Modell nicht gefunden!${NC}"
    echo "Erwarteter Pfad: Vorhersage-Modell/models/baseline_pipeline_final.joblib"
    echo "Stelle sicher, dass du das komplette Repository heruntergeladen hast."
    exit 1
fi

echo ""

# Test-Dateien prüfen
echo "🔍 Prüfe Tool-Dateien..."
MISSING=0
for file in predictor_lib.py quickstart.py; do
    if [ -f "$file" ]; then
        echo -e "   ${GREEN}✓${NC} $file"
    else
        echo -e "   ${RED}✗${NC} $file ${RED}fehlt!${NC}"
        MISSING=1
    fi
done

if [ $MISSING -eq 1 ]; then
    echo -e "${YELLOW}⚠️  Einige Dateien fehlen. Installation fortsetzen? (y/n)${NC}"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""

# Funktionstest
echo "🧪 Führe Funktionstest aus..."
TEST_OUTPUT=$($PYTHON_CMD -c "
from predictor_lib import PredictorModel
predictor = PredictorModel()
result = predictor.predict(
    zone_phase='mid',
    alive_players=25,
    teammates_alive=3,
    height_status='high',
    position_type='edge'
)
print(f\"Call: {result['predicted_call']}, Confidence: {result['confidence']:.0%}\")
" 2>&1)

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Funktionstest erfolgreich!${NC}"
    echo "   Ergebnis: $TEST_OUTPUT"
else
    echo -e "${RED}❌ Funktionstest fehlgeschlagen!${NC}"
    echo "   Fehler: $TEST_OUTPUT"
    exit 1
fi

echo ""
echo "╔═══════════════════════════════════════════════════════╗"
echo "║              ✅ Installation erfolgreich!             ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo ""
echo "🎯 Nächste Schritte:"
echo ""
echo "1. Teste das Quickstart-Beispiel:"
echo "   ${GREEN}$PYTHON_CMD quickstart.py${NC}"
echo ""
echo "2. Probiere alle Features:"
echo "   ${GREEN}$PYTHON_CMD predictor_lib.py${NC}"
echo ""
echo "3. CLI-Tool nutzen:"
echo "   ${GREEN}$PYTHON_CMD predict_cli.py --zone mid --players 20 --team 3 --height high --position edge${NC}"
echo ""
echo "4. Lies die Dokumentation:"
echo "   ${GREEN}cat PYTHON_INTEGRATION.md${NC}"
echo ""
echo "📚 Weitere Infos: LOKALE_INSTALLATION.md"
echo ""
