from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional, List
import pandas as pd
from datetime import datetime
import logging
import os
from contextlib import asynccontextmanager
import asyncio

from train_model import HospitalLoadPredictor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model variable
model_data = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    try:
        model_path = "models/rf_model.joblib"
        if os.path.exists(model_path):
            predictor = HospitalLoadPredictor()
            predictor.load_model(model_path)
            model_data["predictor"] = predictor
            logger.info("Model loaded successfully")
        else:
            logger.warning(f"Model file not found at {model_path}")
            logger.info("Please run train_model.py first")
            # Initialize empty predictor for training endpoint
            model_data["predictor"] = HospitalLoadPredictor()
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        model_data["predictor"] = HospitalLoadPredictor()
    
    # Load historical data for context
    try:
        data_path = "data/hospital_data.csv"
        if os.path.exists(data_path):
            model_data["historical_data"] = pd.read_csv(data_path)
            logger.info(f"Historical data loaded: {len(model_data['historical_data'])} records")
    except Exception as e:
        logger.warning(f"Could not load historical data: {e}")
        model_data["historical_data"] = None
    
    yield
    # Cleanup
    model_data.clear()

app = FastAPI(
    title="MediPulse AI Prediction API",
    description="Hospital patient load forecasting service using AI/ML",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class PredictRequest(BaseModel):
    city: str = Field(..., min_length=1, max_length=100, description="City name")
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    aqi: int = Field(..., ge=0, le=500, description="Air Quality Index")
    temperature: float = Field(..., ge=-50, le=60, description="Temperature in Celsius")
    festival: Optional[str] = Field(None, max_length=50, description="Festival name (optional)")
    outbreak: int = Field(0, ge=0, le=3, description="Outbreak level 0-3")
    
    @validator('date')
    def validate_date(cls, v):
        try:
            datetime.strptime(v, '%Y-%m-%d')
            return v
        except ValueError:
            raise ValueError('Date must be in YYYY-MM-DD format')
    
    @validator('city')
    def validate_city(cls, v):
        valid_cities = ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata']
        if v not in valid_cities:
            raise ValueError(f'City must be one of: {", ".join(valid_cities)}')
        return v

class PredictResponse(BaseModel):
    predicted_patients: int
    confidence_interval: List[int]
    model_confidence: float = Field(..., ge=0, le=1)
    model_version: str = "1.0"
    timestamp: str
    input_summary: dict

class ForecastRequest(BaseModel):
    city: str = Field(..., min_length=1, max_length=100)
    days: int = Field(7, ge=1, le=30, description="Number of days to forecast")
    current_aqi: Optional[int] = Field(150, ge=0, le=500)
    current_temperature: Optional[float] = Field(25, ge=-50, le=60)
    current_outbreak: Optional[int] = Field(0, ge=0, le=3)

class ForecastResponse(BaseModel):
    city: str
    forecast_days: int
    predictions: List[dict]
    summary: dict
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_trained: bool
    data_available: bool
    timestamp: str

class TrainRequest(BaseModel):
    retrain: bool = Field(True, description="Force retraining even if model exists")

# Dependency functions
def get_model():
    if "predictor" not in model_data:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model_data["predictor"]

def get_historical_data():
    return model_data.get("historical_data")

# Route handlers
@app.get("/", response_model=dict)
async def root():
    return {
        "message": "MediPulse AI Prediction API",
        "version": "1.0.0",
        "description": "AI-powered hospital patient load forecasting",
        "endpoints": {
            "predict": "POST /predict - Single prediction",
            "forecast": "POST /forecast - Multi-day forecast", 
            "batch": "POST /predict/batch - Batch predictions",
            "train": "POST /train - Train/retrain model",
            "health": "GET /health - Service health check",
            "docs": "GET /docs - API documentation"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    predictor = model_data.get("predictor")
    historical_data = model_data.get("historical_data")
    
    return HealthResponse(
        status="healthy",
        model_loaded=predictor is not None,
        model_trained=predictor.is_trained if predictor else False,
        data_available=historical_data is not None,
        timestamp=datetime.now().isoformat()
    )

@app.post("/predict", response_model=PredictResponse)
async def predict_patient_load(
    request: PredictRequest,
    predictor=Depends(get_model),
    historical_data=Depends(get_historical_data)
):
    try:
        logger.info(f"Prediction request: {request.city} on {request.date}")
        
        if not predictor.is_trained:
            raise HTTPException(
                status_code=503, 
                detail="Model not trained. Please run training first."
            )
        
        # Make prediction
        result = predictor.predict_single(
            city=request.city,
            date=request.date,
            aqi=request.aqi,
            temperature=request.temperature,
            festival=request.festival or '',
            outbreak=request.outbreak,
            historical_data=historical_data
        )
        
        return PredictResponse(
            predicted_patients=result['predicted_patients'],
            confidence_interval=result['confidence_interval'],
            model_confidence=result['model_confidence'],
            timestamp=datetime.now().isoformat(),
            input_summary={
                "city": request.city,
                "date": request.date,
                "aqi": request.aqi,
                "temperature": request.temperature,
                "has_festival": bool(request.festival),
                "outbreak_level": request.outbreak
            }
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/forecast", response_model=ForecastResponse)
async def forecast_patient_load(
    request: ForecastRequest,
    predictor=Depends(get_model),
    historical_data=Depends(get_historical_data)
):
    try:
        logger.info(f"Forecast request: {request.city} for {request.days} days")
        
        if not predictor.is_trained:
            raise HTTPException(
                status_code=503,
                detail="Model not trained. Please run training first."
            )
        
        # Prepare current conditions
        current_conditions = {
            'aqi': request.current_aqi,
            'temperature': request.current_temperature,
            'outbreak': request.current_outbreak
        }
        
        # Get forecast
        if request.days == 7:
            predictions = predictor.predict_next_7_days(
                city=request.city,
                current_conditions=current_conditions,
                historical_data=historical_data
            )
        else:
            # For other durations, extend the 7-day logic
            predictions = []
            base_date = datetime.now().date()
            
            for i in range(request.days):
                pred_date = base_date + pd.Timedelta(days=i+1)
                result = predictor.predict_single(
                    city=request.city,
                    date=pred_date.strftime('%Y-%m-%d'),
                    aqi=current_conditions['aqi'],
                    temperature=current_conditions['temperature'],
                    outbreak=current_conditions['outbreak'],
                    historical_data=historical_data
                )
                
                predictions.append({
                    'date': pred_date.strftime('%Y-%m-%d'),
                    'day_of_week': pred_date.strftime('%A'),
                    **result
                })
        
        # Calculate summary statistics
        predicted_counts = [p['predicted_patients'] for p in predictions]
        summary = {
            'total_predicted': sum(predicted_counts),
            'daily_average': sum(predicted_counts) / len(predicted_counts),
            'peak_day': max(predictions, key=lambda x: x['predicted_patients']),
            'min_day': min(predictions, key=lambda x: x['predicted_patients']),
            'surge_days': len([p for p in predictions if p['predicted_patients'] > 200])
        }
        
        return ForecastResponse(
            city=request.city,
            forecast_days=request.days,
            predictions=predictions,
            summary=summary,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Forecast error: {e}")
        raise HTTPException(status_code=500, detail=f"Forecast failed: {str(e)}")

@app.post("/predict/batch", response_model=List[PredictResponse])
async def predict_batch(
    requests: List[PredictRequest],
    predictor=Depends(get_model),
    historical_data=Depends(get_historical_data)
):
    if len(requests) > 100:
        raise HTTPException(
            status_code=400, 
            detail="Maximum 100 requests per batch"
        )
    
    if not predictor.is_trained:
        raise HTTPException(
            status_code=503,
            detail="Model not trained. Please run training first."
        )
    
    results = []
    for req in requests:
        try:
            result = predictor.predict_single(
                city=req.city,
                date=req.date,
                aqi=req.aqi,
                temperature=req.temperature,
                festival=req.festival or '',
                outbreak=req.outbreak,
                historical_data=historical_data
            )
            
            results.append(PredictResponse(
                predicted_patients=result['predicted_patients'],
                confidence_interval=result['confidence_interval'],
                model_confidence=result['model_confidence'],
                timestamp=datetime.now().isoformat(),
                input_summary={
                    "city": req.city,
                    "date": req.date,
                    "aqi": req.aqi,
                    "temperature": req.temperature,
                    "has_festival": bool(req.festival),
                    "outbreak_level": req.outbreak
                }
            ))
            
        except Exception as e:
            logger.error(f"Batch prediction error for {req.city}: {e}")
            # Continue with other predictions
            continue
    
    return results

@app.post("/train")
async def train_model(
    request: TrainRequest,
    background_tasks: BackgroundTasks
):
    """Train or retrain the prediction model"""
    try:
        # Check if model exists and training is not forced
        model_path = "models/rf_model.joblib"
        if os.path.exists(model_path) and not request.retrain:
            return {
                "message": "Model already exists. Use retrain=true to force retraining.",
                "model_path": model_path,
                "timestamp": datetime.now().isoformat()
            }
        
        # Check if data exists
        data_path = "data/hospital_data.csv"
        if not os.path.exists(data_path):
            raise HTTPException(
                status_code=400,
                detail="Training data not found. Please run generate_sample_data.py first."
            )
        
        # Start background training
        background_tasks.add_task(train_model_background)
        
        return {
            "message": "Model training started in background",
            "status": "training",
            "estimated_time": "2-5 minutes",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Training initiation error: {e}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

async def train_model_background():
    """Background task for model training"""
    try:
        logger.info("Starting background model training...")
        
        # Load data
        df = pd.read_csv("data/hospital_data.csv")
        
        # Train model
        predictor = HospitalLoadPredictor()
        predictor, cv_results, feature_importance = predictor.train(df)
        
        # Save model
        model_path = "models/rf_model.joblib"
        predictor.save_model(model_path)
        
        # Update global model
        model_data["predictor"] = predictor
        
        logger.info("Background model training completed successfully")
        
    except Exception as e:
        logger.error(f"Background training error: {e}")

# Custom exception handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return HTTPException(status_code=400, detail=str(exc))

@app.exception_handler(FileNotFoundError)
async def file_not_found_handler(request, exc):
    return HTTPException(status_code=404, detail=str(exc))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )