# Predictor API Server

Professional REST API for `best_call` predictions with robust management tools.

## Quick Start

```bash
cd /workspaces/Predictor/Vorhersage-Modell

# Start the server
./manage_api.sh start

# Check status
./manage_api.sh status

# Test endpoints
./manage_api.sh test

# View logs
./manage_api.sh logs

# Stop the server
./manage_api.sh stop
```

## Management Commands

The `manage_api.sh` script provides complete server lifecycle management:

### Start Server
```bash
./manage_api.sh start

# Custom port
PORT=8080 ./manage_api.sh start
```

### Stop Server
```bash
./manage_api.sh stop
```

### Restart Server
```bash
./manage_api.sh restart
```

### Check Status
```bash
./manage_api.sh status
```

### View Logs
```bash
# Last 50 lines
./manage_api.sh logs

# Follow logs in real-time
./manage_api.sh logs -f
```

### Test Endpoints
```bash
./manage_api.sh test
```

## API Endpoints

### 1. Health Check
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "models_available": ["baseline"],
  "timestamp": "2025-11-17T12:00:00.000000+00:00"
}
```

### 2. Models Info
```bash
curl http://localhost:8000/models
```

**Response:**
```json
[
  {
    "name": "baseline",
    "type": "scikit-learn (RF) + ONNX",
    "path": "models/baseline_pipeline_final.joblib",
    "available": true
  }
]
```

### 3. Single Prediction
```bash
curl -X POST http://localhost:8000/predict \
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
    "match_id": "match_001",
    "frame_id": "0001"
  }'
```

**Response:**
```json
{
  "match_id": "match_001",
  "frame_id": "0001",
  "predicted_call": "stick_deadside",
  "probabilities": {
    "stick_deadside": 0.45,
    "play_frontside": 0.25,
    "take_height": 0.15,
    "stabilize_box": 0.10,
    "look_for_refresh": 0.03,
    "drop_low": 0.02
  },
  "confidence": 0.45,
  "model": "baseline"
}
```

### 4. Batch Predictions
```bash
curl -X POST http://localhost:8000/batch-predict \
  -H "Content-Type: application/json" \
  -d '{
    "examples": [
      {
        "zone_phase": "mid",
        "zone_index": 5,
        "alive_players": 30,
        "teammates_alive": 3,
        "storm_edge_dist": 150.5,
        "mats_total": 400,
        "surge_above": 10,
        "height_status": "mid",
        "position_type": "edge",
        "match_id": "match_001",
        "frame_id": "0001"
      }
    ],
    "model": "baseline"
  }'
```

## Interactive Documentation

Open in browser: `http://localhost:8000/docs`

This provides:
- Interactive API testing (Swagger UI)
- Complete endpoint documentation
- Request/response schemas
- Try-it-out functionality

## Architecture

### Components

1. **FastAPI Server** (`src/api_server.py`)
   - Async request handling
   - Pydantic validation
   - Model caching for performance

2. **Management Scripts**
   - `manage_api.sh` - Full lifecycle management
   - `start_api.sh` - Simple startup script

3. **Prediction Modules**
   - `src/predict_best_call.py` - Baseline inference
   - `src/multimodal_model.py` - Multimodal inference

### Model Loading

Models are loaded once at startup and cached in memory:

```python
class ModelCache:
    def __init__(self):
        self.baseline_pipeline = None
        self.onnx_session = None
        self.multimodal_model = None
```

### Error Handling

- **400 Bad Request** - Invalid input data
- **404 Not Found** - Endpoint doesn't exist
- **500 Internal Server Error** - Model/prediction errors
- **501 Not Implemented** - Feature not yet implemented

## Production Deployment

### Docker

```bash
# Build image
docker build -t predictor-api .

# Run container
docker run -p 8000:8000 predictor-api

# Or use docker-compose
docker-compose up
```

### Environment Variables

```bash
PORT=8000          # Server port
HOST=0.0.0.0       # Bind address
LOG_LEVEL=info     # Logging level
```

### Health Checks

```bash
# Kubernetes/Docker health probe
curl -f http://localhost:8000/health || exit 1
```

## Performance

- **Cold start**: ~2-3 seconds (model loading)
- **Single prediction**: ~10-50ms
- **Batch prediction (100)**: ~500ms-2s
- **Concurrent requests**: Supports async handling

## Monitoring

### Logs Location

- **PID file**: `/tmp/predictor_api.pid`
- **Log file**: `/tmp/predictor_api.log`

### Log Format

```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     127.0.0.1:52431 - "GET /health HTTP/1.1" 200 OK
```

## Troubleshooting

### Port Already in Use

```bash
# Find process using port 8000
lsof -ti :8000

# Kill process
kill $(lsof -ti :8000)

# Or use management script
./manage_api.sh stop
```

### Model Not Found

```bash
# Check models directory
ls -lh models/

# Train baseline model if missing
python3 src/train_best_call_baseline.py --data data/call_states_demo.csv

# Export to ONNX
python3 src/export_to_onnx.py
```

### Server Won't Start

```bash
# Check logs
./manage_api.sh logs

# Or directly
tail -f /tmp/predictor_api.log

# Verify Python dependencies
pip install -r requirements.txt
```

### GitHub Codespaces Port Forwarding

If the external URL doesn't work:

1. Open **PORTS** tab in VS Code (bottom panel)
2. Find port **8000**
3. Set visibility to **Public**
4. Click globe icon 🌐 to open in browser
5. Add `/health` or `/docs` to the URL

## Development

### Run with Auto-Reload

```bash
cd /workspaces/Predictor/Vorhersage-Modell
python3 -m uvicorn src.api_server:app --reload --port 8000
```

### Run Tests

```bash
# Full endpoint test
./manage_api.sh test

# Manual test
curl http://localhost:8000/health
```

### Code Changes

After modifying `src/api_server.py`:

```bash
./manage_api.sh restart
```

## Security Notes

- **Production**: Set `HOST=127.0.0.1` for internal-only access
- **CORS**: Currently allows all origins (see `app.add_middleware`)
- **Authentication**: Not implemented (add JWT/API keys for production)
- **Rate Limiting**: Not implemented (add Redis-based limiting for production)

## License

See main project README.
