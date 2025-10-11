"""
Script to fix the ColumnTransformer issue by ensuring it's properly fitted 
and saved with the model.
"""
import joblib
import os
import numpy as np
import pandas as pd
from train_model import HospitalLoadPredictor
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_transformer():
    model_path = "models/rf_model.joblib"
    data_path = "data/hospital_data.csv"
    
    if not os.path.exists(model_path) or not os.path.exists(data_path):
        logger.error(f"Model or data file not found: {model_path}, {data_path}")
        return False
    
    try:
        # Load data
        logger.info("Loading data...")
        df = pd.read_csv(data_path)
        
        # Create predictor
        predictor = HospitalLoadPredictor()
        
        # Train properly
        logger.info("Training model with transformer...")
        df_features = predictor.engineer_features(df)
        feature_cols = predictor.setup_preprocessor(df_features)
        
        # Prepare data (remove rows with NaN values)
        X = df_features[feature_cols].dropna()
        y = df_features.loc[X.index, 'patients_admitted']
        
        # Fit preprocessor
        logger.info("Fitting transformer...")
        X_transformed = predictor.preprocessor.fit_transform(X)
        
        # Train model
        logger.info("Training model...")
        predictor.model.fit(X_transformed, y)
        predictor.is_trained = True
        
        # Save model with transformer
        logger.info(f"Saving model to {model_path}")
        joblib.dump(predictor, model_path)
        
        # Test a prediction
        logger.info("Testing prediction...")
        test_data = {
            'city': 'Mumbai',
            'date': '2024-12-20',
            'aqi': 75,
            'temperature': 25.0,
            'outbreak': 0
        }
        prediction = predictor.predict_single(**test_data)
        logger.info(f"Test prediction result: {prediction}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error fixing transformer: {e}")
        return False

if __name__ == "__main__":
    success = fix_transformer()
    if success:
        logger.info("Transformer fixed successfully!")
    else:
        logger.error("Failed to fix transformer!")