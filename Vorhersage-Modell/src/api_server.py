"""FastAPI REST Server for best_call predictions.

Provides endpoints for single and batch inference using both baseline and multimodal models.

Usage:
    python3 src/api_server.py [--port 8000] [--model {baseline,multimodal}]

Endpoints:
    GET  /health           — Health check
    POST /predict          — Single prediction
    POST /batch-predict    — Batch predictions from JSONL
    GET  /models           — Available models info
"""
import os
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ConfigDict

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import prediction modules
import sys
sys.path.insert(0, os.path.dirname(__file__))
from predict_best_call import predict as baseline_predict
import joblib

# Optional: multimodal imports
try:
    import torch
    from multimodal_model import GameStateImageDataset, MultimodalCNN, evaluate
    MULTIMODAL_AVAILABLE = True
except Exception as e:
    print(f"Warning: Multimodal model not available: {e}")
    MULTIMODAL_AVAILABLE = False

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
BASELINE_PIPELINE_PATH = 'models/baseline_pipeline_final.joblib'
ONNX_MODEL_PATH = 'models/best_call_baseline.onnx'
MULTIMODAL_MODEL_PATH = 'models/multimodal_final.pth'
MAX_BATCH_SIZE = 1000

# ============================================================================
# Pydantic Models for Request/Response Validation
# ============================================================================

class GameStateInput(BaseModel):
    """Single game state for prediction."""
    zone_phase: str
    zone_index: int
    alive_players: int
    teammates_alive: int
    storm_edge_dist: float
    mats_total: int
    surge_above: float
    height_status: str
    position_type: str
    outcome_placement: Optional[int] = None
    outcome_alive_time: Optional[int] = None
    match_id: Optional[str] = None
    frame_id: Optional[str] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "zone_phase": "mid",
                "zone_index": 5,
                "alive_players": 30,
                "teammates_alive": 3,
                "storm_edge_dist": 150.5,
                "mats_total": 400,
                "surge_above": 10,
                "height_status": "mid",
                "position_type": "edge",
                "outcome_placement": 50,
                "outcome_alive_time": 300,
                "match_id": "match_001",
                "frame_id": "frame_001"
            }
        }
    )

class PredictionResponse(BaseModel):
    """Single prediction response."""
    match_id: Optional[str] = None
    frame_id: Optional[str] = None
    predicted_call: str
    probabilities: Dict[str, float]
    confidence: float
    model: str

class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""
    examples: List[GameStateInput] = Field(..., max_length=MAX_BATCH_SIZE)
    model: str = Field("baseline", description="Model to use: 'baseline' or 'multimodal'")

class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    predictions: List[PredictionResponse]
    total: int
    model: str
    timestamp: str

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    models_available: List[str]
    timestamp: str

class ModelInfo(BaseModel):
    """Information about available models."""
    name: str
    type: str
    path: str
    available: bool

# ============================================================================
# Model Loading & Caching
# ============================================================================

class ModelCache:
    """Cache for loaded models to avoid repeated disk I/O."""
    
    def __init__(self):
        self.baseline_pipeline = None
        self.baseline_session = None
        self.multimodal_model = None
        self.multimodal_device = None
    
    def load_baseline(self):
        """Load baseline pipeline and ONNX session."""
        if self.baseline_pipeline is None:
            if not os.path.exists(BASELINE_PIPELINE_PATH):
                raise FileNotFoundError(f"Baseline pipeline not found: {BASELINE_PIPELINE_PATH}")
            self.baseline_pipeline = joblib.load(BASELINE_PIPELINE_PATH)
            logger.info(f"Loaded baseline pipeline from {BASELINE_PIPELINE_PATH}")
        
        if self.baseline_session is None:
            try:
                import onnxruntime as ort
                if os.path.exists(ONNX_MODEL_PATH):
                    self.baseline_session = ort.InferenceSession(
                        ONNX_MODEL_PATH,
                        providers=['CPUExecutionProvider']
                    )
                    logger.info(f"Loaded ONNX model from {ONNX_MODEL_PATH}")
                else:
                    logger.warning(f"ONNX model not found: {ONNX_MODEL_PATH}, will use Python classifier")
            except Exception as e:
                logger.warning(f"Could not load ONNX model: {e}, will use Python classifier")
        
        return self.baseline_pipeline, self.baseline_session
    
    def load_multimodal(self):
        """Load multimodal PyTorch model."""
        if not MULTIMODAL_AVAILABLE:
            raise RuntimeError("Multimodal model not available (PyTorch/dependencies missing)")
        
        if self.multimodal_model is None:
            if not os.path.exists(MULTIMODAL_MODEL_PATH):
                raise FileNotFoundError(f"Multimodal model not found: {MULTIMODAL_MODEL_PATH}")
            
            self.multimodal_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # For now, create a dummy model to load state (full multimodal inference requires dataset context)
            # This is a placeholder; full implementation would require image loading logic
            logger.warning("Multimodal inference requires image directory context; returning placeholder")
            self.multimodal_model = "placeholder"
        
        return self.multimodal_model
    
    def is_baseline_available(self) -> bool:
        return os.path.exists(BASELINE_PIPELINE_PATH)
    
    def is_multimodal_available(self) -> bool:
        return MULTIMODAL_AVAILABLE and os.path.exists(MULTIMODAL_MODEL_PATH)

model_cache = ModelCache()

# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Predictor API",
    description="REST API for best_call predictions (baseline + multimodal)",
    version="1.0.0"
)

# CORS middleware for external access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for web interface
static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir), html=True), name="static")

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint."""
    available_models = []
    if model_cache.is_baseline_available():
        available_models.append("baseline")
    if model_cache.is_multimodal_available():
        available_models.append("multimodal")
    
    return HealthResponse(
        status="healthy",
        models_available=available_models,
        timestamp=datetime.now(timezone.utc).isoformat()
    )

@app.get("/models", response_model=List[ModelInfo], tags=["System"])
async def get_models():
    """List available models and their status."""
    return [
        ModelInfo(
            name="baseline",
            type="scikit-learn (RF) + ONNX",
            path=BASELINE_PIPELINE_PATH,
            available=model_cache.is_baseline_available()
        ),
        ModelInfo(
            name="multimodal",
            type="PyTorch (CNN + Tabular Fusion)",
            path=MULTIMODAL_MODEL_PATH,
            available=model_cache.is_multimodal_available()
        )
    ]

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(example: GameStateInput, model: str = "baseline"):
    """
    Single prediction endpoint.
    
    Args:
        example: Game state features
        model: Model to use ('baseline' or 'multimodal', default: 'baseline')
    
    Returns:
        Prediction with class label and probabilities
    """
    try:
        # Convert Pydantic model to dict
        example_dict = example.dict()
        
        if model == "baseline":
            # Load baseline pipeline and ONNX session
            pipeline, session = model_cache.load_baseline()
            
            # Call baseline predict
            pred_label, probs, labels = baseline_predict(example_dict, pipeline=pipeline, sess=session)
            
            # Build probability dict
            prob_dict = {lab: float(p) for lab, p in zip(labels, probs)}
            
            # Confidence is the max probability
            confidence = float(max(probs))
            
            return PredictionResponse(
                match_id=example.match_id,
                frame_id=example.frame_id,
                predicted_call=pred_label,
                probabilities=prob_dict,
                confidence=confidence,
                model="baseline"
            )
        
        elif model == "multimodal":
            if not model_cache.is_multimodal_available():
                raise HTTPException(
                    status_code=503,
                    detail="Multimodal model not available. Requires PyTorch and image directory."
                )
            # Placeholder: multimodal inference requires full context (images, dataset)
            raise HTTPException(
                status_code=501,
                detail="Multimodal single prediction not yet implemented. Use /batch-predict with images directory."
            )
        
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model: {model}. Choose 'baseline' or 'multimodal'."
            )
    
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Model not found: {str(e)}")
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/batch-predict", response_model=BatchPredictionResponse, tags=["Prediction"])
async def batch_predict(request: BatchPredictionRequest):
    """
    Batch prediction endpoint.
    
    Args:
        request: Batch of game states + model choice
    
    Returns:
        List of predictions
    """
    try:
        if len(request.examples) > MAX_BATCH_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"Batch size {len(request.examples)} exceeds maximum {MAX_BATCH_SIZE}"
            )
        
        predictions = []
        
        if request.model == "baseline":
            pipeline = model_cache.load_baseline()
            
            for example in request.examples:
                example_dict = example.dict()
                pred_label, probs, labels = baseline_predict(example_dict, pipeline=pipeline)
                prob_dict = {lab: float(p) for lab, p in zip(labels, probs)}
                confidence = float(max(probs))
                
                predictions.append(PredictionResponse(
                    match_id=example.match_id,
                    frame_id=example.frame_id,
                    predicted_call=pred_label,
                    probabilities=prob_dict,
                    confidence=confidence,
                    model="baseline"
                ))
        
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model: {request.model}. Choose 'baseline' or 'multimodal'."
            )
        
        return BatchPredictionResponse(
            predictions=predictions,
            total=len(predictions),
            model=request.model,
            timestamp=datetime.utcnow().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8000, help='Port to run API server on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload on code changes')
    parser.add_argument('--workers', type=int, default=1, help='Number of worker processes')
    args = parser.parse_args()
    
    logger.info(f"Starting API server on {args.host}:{args.port}")
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers
    )
