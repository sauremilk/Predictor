#!/bin/bash
# Robustes Startup-Skript für den API-Server

set -e

# Farbige Ausgabe
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}Predictor API Server Startup${NC}"
echo -e "${GREEN}================================${NC}\n"

# Arbeitsverzeichnis setzen
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo -e "${YELLOW}Working directory:${NC} $PWD"

# Prüfe Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}ERROR: python3 not found${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} Python3 found: $(python3 --version)"

# Prüfe Requirements
if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}ERROR: requirements.txt not found${NC}"
    exit 1
fi

# Prüfe ob Modelle existieren
if [ ! -f "models/baseline_pipeline_final.joblib" ]; then
    echo -e "${YELLOW}WARNING: models/baseline_pipeline_final.joblib not found${NC}"
    echo -e "${YELLOW}You may need to train a model first${NC}"
fi

if [ ! -f "models/best_call_baseline.onnx" ]; then
    echo -e "${YELLOW}WARNING: models/best_call_baseline.onnx not found${NC}"
    echo -e "${YELLOW}You may need to export the model first${NC}"
fi

# Port-Parameter
PORT=${1:-8000}
HOST=${2:-0.0.0.0}

echo -e "\n${YELLOW}Configuration:${NC}"
echo -e "  Host: $HOST"
echo -e "  Port: $PORT"

# Prüfe ob Port bereits belegt
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo -e "\n${RED}ERROR: Port $PORT is already in use${NC}"
    echo -e "Stop the process with: ${YELLOW}kill \$(lsof -t -i:$PORT)${NC}"
    exit 1
fi

echo -e "\n${GREEN}Starting API server...${NC}\n"

# Starte Server mit uvicorn direkt (robuster als via api_server.py)
exec python3 -m uvicorn src.api_server:app \
    --host "$HOST" \
    --port "$PORT" \
    --log-level info \
    --access-log
