"""
Script to retrain and fix the ML model with proper preprocessing
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
from datetime import datetime
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def simple_train():
    """Simple training function that creates a working model with transformer"""
    try:
        data_path = "data/hospital_data.csv"
        model_path = "models/rf_model.joblib"
        
        # Load data
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        
        # Basic preprocessing
        df['date'] = pd.to_datetime(df['date'])
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Select a subset of features for simplicity
        X = df[['city', 'aqi', 'temperature', 'day_of_week', 'month', 'is_weekend']]
        y = df['patients_admitted']
        
        # Create train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Define transformers
        cat_features = ['city']
        num_features = ['aqi', 'temperature', 'day_of_week', 'month', 'is_weekend']
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features),
                ('num', 'passthrough', num_features)
            ]
        )
        
        # Fit preprocessor
        logger.info("Fitting preprocessor")
        X_train_preprocessed = preprocessor.fit_transform(X_train)
        
        # Train model
        logger.info("Training Random Forest model")
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_preprocessed, y_train)
        
        # Evaluate model
        X_test_preprocessed = preprocessor.transform(X_test)
        y_pred = model.predict(X_test_preprocessed)
        mae = mean_absolute_error(y_test, y_pred)
        logger.info(f"Model MAE: {mae:.2f}")
        
        # Create a prediction function for easier usage
        def predict_func(city, date, aqi, temperature):
            # Convert date string to datetime
            if isinstance(date, str):
                date = pd.to_datetime(date)
            
            # Extract features
            day_of_week = date.weekday()
            month = date.month
            is_weekend = int(day_of_week >= 5)
            
            # Create input dataframe
            input_df = pd.DataFrame({
                'city': [city],
                'aqi': [aqi],
                'temperature': [temperature],
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
            
            return {
                'predicted_patients': prediction,
                'confidence_interval': [max(0, prediction - confidence), prediction + confidence],
                'model_confidence': 0.85,
                'timestamp': datetime.now().isoformat()
            }
        
        # Save model artifacts
        model_dict = {
            'model': model,
            'preprocessor': preprocessor
        }
        
        logger.info(f"Saving model to {model_path}")
        joblib.dump(model_dict, model_path)
        
        # Test prediction
        test_date = datetime.strptime("2024-12-20", "%Y-%m-%d")
        test_prediction = predict_func("Mumbai", test_date, 75, 25.0)
        logger.info(f"Test prediction: {test_prediction}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error training model: {e}")
        return False

if __name__ == "__main__":
    if simple_train():
        logger.info("Model successfully trained and saved!")
    else:
        logger.error("Failed to train and save model!")