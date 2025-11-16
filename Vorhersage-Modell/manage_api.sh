#!/bin/bash
# API Server Management Tool

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PID_FILE="/tmp/predictor_api.pid"
LOG_FILE="/tmp/predictor_api.log"
PORT=${PORT:-8000}

# Farbige Ausgabe
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

function show_status() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            echo -e "${GREEN}✓ Server is RUNNING${NC} (PID: $PID)"
            echo -e "  Port: $PORT"
            echo -e "  URL: http://localhost:$PORT"
            echo -e "  Logs: $LOG_FILE"
            return 0
        else
            echo -e "${YELLOW}⚠ PID file exists but process is not running${NC}"
            rm -f "$PID_FILE"
            return 1
        fi
    else
        echo -e "${RED}✗ Server is NOT RUNNING${NC}"
        return 1
    fi
}

function start_server() {
    if show_status > /dev/null 2>&1; then
        echo -e "${YELLOW}Server is already running${NC}"
        show_status
        return 0
    fi

    echo -e "${BLUE}Starting Predictor API Server...${NC}\n"
    
    # Starte Server im Hintergrund
    nohup python3 -m uvicorn src.api_server:app \
        --host 0.0.0.0 \
        --port "$PORT" \
        --log-level info \
        > "$LOG_FILE" 2>&1 &
    
    PID=$!
    echo $PID > "$PID_FILE"
    
    # Warte kurz und prüfe ob gestartet
    sleep 2
    
    if ps -p "$PID" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Server started successfully${NC}"
        show_status
        echo -e "\n${BLUE}Test the server:${NC}"
        echo -e "  curl http://localhost:$PORT/health"
        echo -e "\n${BLUE}View API docs:${NC}"
        echo -e "  Open: http://localhost:$PORT/docs"
    else
        echo -e "${RED}✗ Failed to start server${NC}"
        echo -e "Check logs: tail -f $LOG_FILE"
        rm -f "$PID_FILE"
        return 1
    fi
}

function stop_server() {
    if ! show_status > /dev/null 2>&1; then
        echo -e "${YELLOW}Server is not running${NC}"
        return 0
    fi
    
    PID=$(cat "$PID_FILE")
    echo -e "${BLUE}Stopping server (PID: $PID)...${NC}"
    
    kill "$PID" 2>/dev/null || true
    
    # Warte auf sauberes Herunterfahren
    for i in {1..10}; do
        if ! ps -p "$PID" > /dev/null 2>&1; then
            break
        fi
        sleep 0.5
    done
    
    # Force kill falls nötig
    if ps -p "$PID" > /dev/null 2>&1; then
        echo -e "${YELLOW}Force killing server...${NC}"
        kill -9 "$PID" 2>/dev/null || true
    fi
    
    rm -f "$PID_FILE"
    echo -e "${GREEN}✓ Server stopped${NC}"
}

function restart_server() {
    echo -e "${BLUE}Restarting server...${NC}\n"
    stop_server
    sleep 1
    start_server
}

function show_logs() {
    if [ ! -f "$LOG_FILE" ]; then
        echo -e "${RED}No log file found${NC}"
        return 1
    fi
    
    if [ "$1" = "-f" ] || [ "$1" = "--follow" ]; then
        echo -e "${BLUE}Following logs (Ctrl+C to stop)...${NC}\n"
        tail -f "$LOG_FILE"
    else
        echo -e "${BLUE}Last 50 lines of logs:${NC}\n"
        tail -n 50 "$LOG_FILE"
    fi
}

function test_server() {
    if ! show_status > /dev/null 2>&1; then
        echo -e "${RED}Server is not running. Start it first with: $0 start${NC}"
        return 1
    fi
    
    echo -e "${BLUE}Testing server endpoints...${NC}\n"
    
    # Test health
    echo -e "${YELLOW}1. Health Check:${NC}"
    curl -s "http://localhost:$PORT/health" | python3 -m json.tool || echo -e "${RED}Failed${NC}"
    
    echo -e "\n${YELLOW}2. Models Info:${NC}"
    curl -s "http://localhost:$PORT/models" | python3 -m json.tool || echo -e "${RED}Failed${NC}"
    
    echo -e "\n${YELLOW}3. Sample Prediction:${NC}"
    curl -s -X POST "http://localhost:$PORT/predict" \
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
            "match_id": "test_001",
            "frame_id": "0001"
        }' | python3 -m json.tool || echo -e "${RED}Failed${NC}"
    
    echo -e "\n${GREEN}✓ Tests completed${NC}"
}

function show_help() {
    cat << EOF
${GREEN}Predictor API Server Management${NC}

${YELLOW}Usage:${NC}
    $0 <command> [options]

${YELLOW}Commands:${NC}
    start       Start the API server
    stop        Stop the API server
    restart     Restart the API server
    status      Show server status
    logs [-f]   Show logs (use -f to follow)
    test        Run endpoint tests
    help        Show this help message

${YELLOW}Environment Variables:${NC}
    PORT        Server port (default: 8000)

${YELLOW}Examples:${NC}
    $0 start                    # Start server on port 8000
    PORT=8080 $0 start          # Start server on port 8080
    $0 logs -f                  # Follow logs in real-time
    $0 test                     # Test all endpoints

${YELLOW}Files:${NC}
    PID:  $PID_FILE
    Logs: $LOG_FILE

EOF
}

# Main
case "${1:-}" in
    start)
        start_server
        ;;
    stop)
        stop_server
        ;;
    restart)
        restart_server
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs "$2"
        ;;
    test)
        test_server
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo -e "${RED}Unknown command: ${1:-}${NC}\n"
        show_help
        exit 1
        ;;
esac
