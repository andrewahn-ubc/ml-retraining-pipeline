from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from typing import List
from api.model_manager import ModelManager
import logging
import time
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import numpy as np
from contextlib import asynccontextmanager

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter("ml_api_requests_total", "Total API requests", ["endpoint", "status"])
REQUEST_LATENCY = Histogram("ml_api_requests_latency_seconds", "Request Latency", ["endpoint"])
PREDICTION_COUNT = Counter("ml_api_predictions_total", "Total predictions made")

# create server 
model_manager = ModelManager()
start_time = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global start_time

    # startup
    logger.info("loading stock prediction model...")
    model_manager.load_model()
    start_time = time.time()
    logger.info(f"model loaded: version {model_manager.get_version()}")

    yield

    # shutdown
    logger.info("application shutting down")

app = FastAPI(
    "Automated ML Retraining Pipeline", 
    version = "1.0.0",
    lifespan=lifespan
)

class PredictionRequest(BaseModel):
    # Nested list allows for batch requests
    features: List[List[float]]

class PredictionResponse(BaseModel):
    predictions: List[str] # 'up' or 'down'
    probabilities: List[List[float]]
    model_version: str
    latency_ms: float

class HealthResponse(BaseModel):
    status: str
    model_version: str
    model_accuracy: float
    last_trained: str
    uptime_seconds: float

@app.get("/health", response_class=HealthResponse)
async def get_health():
    REQUEST_COUNT.labels(endpoint='/health', status='success').inc()

    return HealthResponse(
        status = "healthy",
        model_version = model_manager.get_version(),
        model_accuracy = model_manager.get_accuracy(),
        last_trained = model_manager.get_trained_date(),
        uptime_seconds = (time.time() - start_time)
    )

@app.post("/predict", response_class=PredictionResponse) 
async def predict(request: PredictionRequest):
    # predict whether the S&P 500 will move up or down on the next trading day
    # Expected features:
        # SMA_10: 10-day simple moving average
        # SMA_5: 5-day simple moving average
        # RSI: Relative Strength Index
        # volume_change: % change in volume
        # price_change: % change in price
    
    start = time.time()

    try:
        X = np.array(request.features)  # convert list of ints to numpy arrays
        if X.shape[1] != 5:
            raise HTTPException(400, "Samples have an invalid number of features (not 5 like we expected)")
        
        # compute predictions
        predictions = model_manager.predict(X)
        labels = ['up' if p == 1 else 'down' for p in predictions]

        # compute probabilities
        probabilities = model_manager.predict_proba(X)

        latency = (time.time() - start) * 1000

        # update metrics
        REQUEST_COUNT.labels('/predict', 'success').inc() 
        REQUEST_LATENCY.labels('/predict').observe(latency) 
        PREDICTION_COUNT.labels().inc(len(predictions))

        logger.info(f"predicted {len(predictions)} samples in {latency:.2f} ms")

        return PredictionResponse(
            predictions = labels,
            probabilities = probabilities.tolist(),
            model_version = model_manager.get_version(),
            latency_ms = latency
        )
    except Exception as e:
        REQUEST_COUNT.labels(endpoint="/predict", status="error").inc()
        logger.error(f"prediction error: {str(e)}")
        raise HTTPException(500, str(e))
    
@app.get('/metrics')
async def metrics():
    # return Prometheus metrics
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

@app.post('/reload')
async def reload_model():
    # reload model (after retraining!)
    try:
        model_manager.load_model()
        REQUEST_COUNT.labels(endpoint="/reload", status="success").inc()
        return {
            "status": "success",
            "msssage": f"Successfully reloaded model version {model_manager.get_version()}.",
            "accuracy": model_manager.get_accuracy(),
            "trained": model_manager.get_trained_date()
        }
    except Exception as e:
        REQUEST_COUNT.labels(endpoint="/reload", status="error").inc()
        raise HTTPException(400, str(e))
    
@app.get('/')
async def root():
    # return API info
    return {
        "name": "Stock Prediction API",
        "model": model_manager.get_version(),
        "endpoints": {
            "predict": "/predict",
            "reload": "/reload",
            "health": "/health",
            "metrics": "/metrics"
        }
    }