"""
Simple prediction API that works with minimal dependencies
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
from datetime import datetime
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="Simple Patient Load Predictor")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
try:
    model_path = "models/simple_model.joblib"
    model_data = joblib.load(model_path)
    model = model_data['model']
    features = model_data['features']
    logger.info(f"Model loaded successfully from {model_path}")
    logger.info(f"Model features: {features}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None
    features = []

# Pydantic models
class PredictionInput(BaseModel):
    city: str = Field(..., description="City name")
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    aqi: float = Field(..., description="Air Quality Index (0-500)")
    temperature: float = Field(..., description="Temperature in Celsius")

class PredictionOutput(BaseModel):
    predicted_patients: int
    confidence_interval: list
    model_confidence: float
    timestamp: str

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "features": features,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictionOutput)
async def predict(data: PredictionInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Extract features we need (just aqi and temperature)
        aqi = float(data.aqi)
        temperature = float(data.temperature)
        
        # Make prediction
        prediction = model.predict([[aqi, temperature]])[0]
        prediction_int = int(round(prediction))
        
        # Create confidence interval
        ci = [max(0, prediction_int - 20), prediction_int + 20]
        
        # Return result
        return {
            "predicted_patients": prediction_int,
            "confidence_interval": ci,
            "model_confidence": 0.85,
            "timestamp": datetime.now().isoformat()
        }
        
    except ValueError as e:
        logger.error(f"Value error: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)