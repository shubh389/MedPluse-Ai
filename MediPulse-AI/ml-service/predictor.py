"""
Basic prediction service with proper error handling
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from datetime import datetime
import logging
import numpy as np
import joblib
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="Simple Patient Load Predictor")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class PredictionInput(BaseModel):
    city: str
    date: str
    aqi: float
    temperature: float

class PredictionOutput(BaseModel):
    predicted_patients: int
    confidence_interval: list
    model_confidence: float
    timestamp: str

# Load model on startup
try:
    model_path = "models/rf_model.joblib"
    model_data = joblib.load(model_path)
    model = model_data['model']
    preprocessor = model_data['preprocessor']
    logger.info(f"Model loaded from {model_path}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None
    preprocessor = None

@app.get("/")
async def root():
    return {"message": "MediPulse AI Patient Load Prediction Service"}

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictionOutput)
async def predict(data: PredictionInput):
    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure the model is properly trained."
        )
    
    try:
        # Parse date
        date = pd.to_datetime(data.date)
        
        # Extract features
        day_of_week = date.weekday()
        month = date.month
        is_weekend = 1 if day_of_week >= 5 else 0
        
        # Create input DataFrame
        input_data = pd.DataFrame({
            'city': [data.city],
            'aqi': [data.aqi],
            'temperature': [data.temperature],
            'day_of_week': [day_of_week],
            'month': [month],
            'is_weekend': [is_weekend]
        })
        
        # Transform input data
        try:
            input_transformed = preprocessor.transform(input_data)
        except Exception as e:
            logger.error(f"Preprocessing error: {e}")
            raise HTTPException(
                status_code=400, 
                detail=f"Error preprocessing input data: {str(e)}"
            )
        
        # Make prediction
        prediction = model.predict(input_transformed)[0]
        prediction_rounded = int(round(prediction))
        
        # Add confidence interval
        margin = 20  # +/- 20 patients as confidence interval
        lower_bound = max(0, prediction_rounded - margin)
        upper_bound = prediction_rounded + margin
        
        # Return response
        return PredictionOutput(
            predicted_patients=prediction_rounded,
            confidence_interval=[lower_bound, upper_bound],
            model_confidence=0.85,  # Fixed confidence for now
            timestamp=datetime.now().isoformat()
        )
        
    except ValueError as e:
        logger.error(f"Value error in prediction: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("predictor:app", host="0.0.0.0", port=8000, reload=False)