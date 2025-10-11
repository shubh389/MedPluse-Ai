"""
Create a very simple model for prediction without complex preprocessing
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_simple_model():
    try:
        # Load data
        logger.info("Loading data")
        df = pd.read_csv("data/hospital_data.csv")
        
        # Simplify by just using numerical features
        features = ['aqi', 'temperature']
        X = df[features]
        y = df['patients_admitted']
        
        # Train a simple model
        logger.info("Training model")
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Create a simple prediction function
        def predict_func(aqi, temperature):
            # Make prediction directly
            prediction = model.predict([[aqi, temperature]])[0]
            prediction_int = int(round(prediction))
            
            # Simple confidence interval
            ci = [max(0, prediction_int - 20), prediction_int + 20]
            
            return {
                'predicted_patients': prediction_int,
                'confidence_interval': ci,
                'model_confidence': 0.85
            }
        
        # Save the model
        logger.info("Saving model")
        model_dict = {
            'model': model,
            'features': features
        }
        joblib.dump(model_dict, 'models/simple_model.joblib')
        
        # Test prediction
        result = predict_func(75, 25.0)
        logger.info(f"Test prediction: {result}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error creating model: {e}")
        return False

if __name__ == "__main__":
    if create_simple_model():
        logger.info("Simple model created successfully!")
    else:
        logger.error("Failed to create simple model")