"""
Updated simple prediction endpoint for ML service
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="MediPulse AI Prediction Service")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:5000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
try:
    model_dict = joblib.load('models/rf_model.joblib')
    model = model_dict['model']
    preprocessor = model_dict['preprocessor']
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None
    preprocessor = None

# Pydantic models
class PredictRequest(BaseModel):
    city: str
    date: str
    aqi: float
    temperature: float

class PredictResponse(BaseModel):
    predicted_patients: int
    confidence_interval: list
    model_confidence: float
    timestamp: str

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_trained": model is not None,
        "data_available": True,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert date string to datetime
        date = pd.to_datetime(request.date)
        
        # Extract features
        day_of_week = date.weekday()
        month = date.month
        is_weekend = int(day_of_week >= 5)
        
        # Create input dataframe
        input_df = pd.DataFrame({
            'city': [request.city],
            'aqi': [request.aqi],
            'temperature': [request.temperature],
            'day_of_week': [day_of_week],
            'month': [month],
            'is_weekend': [is_weekend]
        })
        
        # Preprocess
        input_preprocessed = preprocessor.transform(input_df)
        
        # Predict
        prediction = model.predict(input_preprocessed)[0]
        
        # Round to nearest integer and create confidence interval
        prediction = round(prediction)
        confidence = 20  # +/- 20 patients
        
        return PredictResponse(
            predicted_patients=prediction,
            confidence_interval=[max(0, prediction - confidence), prediction + confidence],
            model_confidence=0.85,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)