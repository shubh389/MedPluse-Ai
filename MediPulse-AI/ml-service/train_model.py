import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
from datetime import datetime, timedelta
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HospitalLoadPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=300,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        self.preprocessor = None
        self.feature_columns = []
        self.is_trained = False
    
    def engineer_features(self, df):
        """Engineer temporal and contextual features"""
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['city', 'date'])
        
        # Temporal features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['day_of_year'] = df['date'].dt.dayofyear
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        
        # Lag features (by city to avoid leakage)
        df['patients_lag_1'] = df.groupby('city')['patients_admitted'].shift(1)
        df['patients_lag_7'] = df.groupby('city')['patients_admitted'].shift(7)
        df['patients_lag_30'] = df.groupby('city')['patients_admitted'].shift(30)
        
        # Rolling averages
        df['patients_ma_7'] = df.groupby('city')['patients_admitted'].rolling(7, min_periods=1).mean().values
        df['patients_ma_30'] = df.groupby('city')['patients_admitted'].rolling(30, min_periods=1).mean().values
        df['aqi_ma_3'] = df.groupby('city')['aqi'].rolling(3, min_periods=1).mean().values
        df['temp_ma_7'] = df.groupby('city')['temperature'].rolling(7, min_periods=1).mean().values
        
        # Festival encoding
        major_festivals = ['Diwali', 'Holi', 'Eid', 'Christmas', 'Dussehra', 'New Year']
        df['festival'] = df['festival'].fillna('')
        df['is_festival'] = (df['festival'] != '').astype(int)
        df['is_major_festival'] = df['festival'].isin(major_festivals).astype(int)
        
        # Festival category
        df['festival_category'] = df['festival'].apply(
            lambda x: x if x in major_festivals else ('Other' if x != '' else 'None')
        )
        
        # Season mapping
        df['season'] = df['month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
        })
        
        # Interaction features
        df['aqi_temp_interaction'] = df['aqi'] * df['temperature']
        df['festival_aqi_interaction'] = df['is_festival'] * df['aqi']
        df['outbreak_aqi_interaction'] = df['outbreak'] * df['aqi']
        df['weekend_festival_interaction'] = df['is_weekend'] * df['is_festival']
        
        # AQI categories
        df['aqi_category'] = pd.cut(df['aqi'], 
                                   bins=[0, 50, 100, 150, 200, 300, 500], 
                                   labels=['Good', 'Moderate', 'Unhealthy_Sensitive', 
                                          'Unhealthy', 'Very_Unhealthy', 'Hazardous'])
        
        # Temperature categories
        df['temp_category'] = pd.cut(df['temperature'],
                                    bins=[-50, 10, 20, 30, 40, 60],
                                    labels=['Cold', 'Cool', 'Moderate', 'Hot', 'Very_Hot'])
        
        return df
    
    def setup_preprocessor(self, df):
        """Setup preprocessing pipeline"""
        categorical_features = ['festival_category', 'season', 'aqi_category', 'temp_category']
        numerical_features = [
            'aqi', 'temperature', 'outbreak', 'day_of_week', 'is_weekend', 
            'day_of_year', 'month', 'quarter', 'patients_lag_1', 'patients_lag_7', 
            'patients_lag_30', 'patients_ma_7', 'patients_ma_30', 'aqi_ma_3', 
            'temp_ma_7', 'is_festival', 'is_major_festival', 'aqi_temp_interaction',
            'festival_aqi_interaction', 'outbreak_aqi_interaction', 
            'weekend_festival_interaction'
        ]
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features),
                ('num', 'passthrough', numerical_features)
            ]
        )
        
        return categorical_features + numerical_features
    
    def train(self, df):
        """Train the model with cross-validation"""
        logger.info("Starting model training...")
        
        # Engineer features
        df_features = self.engineer_features(df)
        feature_cols = self.setup_preprocessor(df_features)
        
        # Prepare data (remove rows with NaN values)
        X = df_features[feature_cols].dropna()
        y = df_features.loc[X.index, 'patients_admitted']
        
        logger.info(f"Training data shape: {X.shape}")
        logger.info(f"Target distribution: mean={y.mean():.1f}, std={y.std():.1f}")
        
        # Fit preprocessor
        X_transformed = self.preprocessor.fit_transform(X)
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_transformed)):
            X_train_fold, X_val_fold = X_transformed[train_idx], X_transformed[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train fold model
            fold_model = RandomForestRegressor(
                n_estimators=300, max_depth=15, min_samples_split=5, 
                random_state=42, n_jobs=-1
            )
            fold_model.fit(X_train_fold, y_train_fold)
            
            # Validate
            y_pred = fold_model.predict(X_val_fold)
            mae = mean_absolute_error(y_val_fold, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred))
            cv_scores.append({'fold': fold, 'mae': mae, 'rmse': rmse})
            
            logger.info(f"Fold {fold}: MAE={mae:.2f}, RMSE={rmse:.2f}")
        
        # Summary CV results
        mean_mae = np.mean([s['mae'] for s in cv_scores])
        std_mae = np.std([s['mae'] for s in cv_scores])
        mean_rmse = np.mean([s['rmse'] for s in cv_scores])
        
        logger.info(f"Cross-validation Results:")
        logger.info(f"MAE: {mean_mae:.2f} ± {std_mae:.2f}")
        logger.info(f"RMSE: {mean_rmse:.2f}")
        
        # Final training on all data
        self.model.fit(X_transformed, y)
        
        # Feature importance analysis
        feature_names = self.get_feature_names()
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info(f"Top 10 Most Important Features:")
        for idx, row in importance_df.head(10).iterrows():
            logger.info(f"{row['feature']}: {row['importance']:.4f}")
        
        self.is_trained = True
        return self, cv_scores, importance_df
    
    def get_feature_names(self):
        """Get feature names after preprocessing"""
        if self.preprocessor is None:
            return []
        
        cat_features = self.preprocessor.named_transformers_['cat'].get_feature_names_out()
        num_features = self.preprocessor.named_transformers_['num']
        
        # Handle numerical features (passthrough)
        if hasattr(num_features, 'get_feature_names_out'):
            num_feature_names = num_features.get_feature_names_out()
        else:
            # For passthrough, use the input feature names
            num_feature_names = [
                'aqi', 'temperature', 'outbreak', 'day_of_week', 'is_weekend', 
                'day_of_year', 'month', 'quarter', 'patients_lag_1', 'patients_lag_7', 
                'patients_lag_30', 'patients_ma_7', 'patients_ma_30', 'aqi_ma_3', 
                'temp_ma_7', 'is_festival', 'is_major_festival', 'aqi_temp_interaction',
                'festival_aqi_interaction', 'outbreak_aqi_interaction', 
                'weekend_festival_interaction'
            ]
        
        return list(cat_features) + list(num_feature_names)
    
    def predict_single(self, city, date, aqi, temperature, festival='', outbreak=0, historical_data=None):
        """Make a single prediction"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Create input dataframe
        input_data = {
            'city': [city],
            'date': [pd.to_datetime(date)],
            'aqi': [aqi],
            'temperature': [temperature],
            'festival': [festival],
            'outbreak': [outbreak]
        }
        
        # If historical data provided, use for lag features
        if historical_data is not None:
            city_data = historical_data[historical_data['city'] == city].copy()
            if len(city_data) > 0:
                city_data = city_data.sort_values('date')
                # Use recent values for lag features
                input_data['patients_admitted'] = [city_data['patients_admitted'].iloc[-1]]
        else:
            # Use average as fallback
            input_data['patients_admitted'] = [100]  # Will be used for lag calculation
        
        df = pd.DataFrame(input_data)
        df_features = self.engineer_features(df)
        
        # Handle missing lag features for prediction
        for col in ['patients_lag_1', 'patients_lag_7', 'patients_lag_30', 'patients_ma_7', 'patients_ma_30']:
            if col not in df_features.columns or df_features[col].isna().all():
                df_features[col] = 100  # Default value
        
        for col in ['aqi_ma_3', 'temp_ma_7']:
            if col not in df_features.columns or df_features[col].isna().all():
                if 'aqi' in col:
                    df_features[col] = df_features['aqi']
                else:
                    df_features[col] = df_features['temperature']
        
        # Get feature columns used in training
        feature_cols = self.setup_preprocessor(df_features)
        X = df_features[feature_cols]
        
        # Transform and predict
        X_transformed = self.preprocessor.transform(X)
        prediction = self.model.predict(X_transformed)[0]
        
        # Calculate prediction interval (rough estimate using std)
        # In production, you'd use quantile regression or ensemble methods
        base_std = prediction * 0.15  # Assume 15% coefficient of variation
        lower_bound = max(0, prediction - 1.96 * base_std)
        upper_bound = prediction + 1.96 * base_std
        
        return {
            'predicted_patients': int(round(prediction)),
            'confidence_interval': [int(round(lower_bound)), int(round(upper_bound))],
            'model_confidence': min(1.0, max(0.5, 1 - (base_std / prediction) if prediction > 0 else 0.5))
        }
    
    def predict_next_7_days(self, city, base_date=None, current_conditions=None, historical_data=None):
        """Predict patient load for next 7 days"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if base_date is None:
            base_date = datetime.now().date()
        elif isinstance(base_date, str):
            base_date = pd.to_datetime(base_date).date()
        
        predictions = []
        
        # Default conditions if not provided
        if current_conditions is None:
            current_conditions = {
                'aqi': 150,
                'temperature': 25,
                'outbreak': 0
            }
        
        for i in range(7):
            pred_date = base_date + timedelta(days=i+1)
            
            try:
                # For simplicity, assume stable environmental conditions
                # In production, you'd integrate with weather/AQI APIs
                result = self.predict_single(
                    city=city,
                    date=pred_date.strftime('%Y-%m-%d'),
                    aqi=current_conditions.get('aqi', 150),
                    temperature=current_conditions.get('temperature', 25),
                    festival='',  # Would check festival calendar
                    outbreak=current_conditions.get('outbreak', 0),
                    historical_data=historical_data
                )
                
                predictions.append({
                    'date': pred_date.strftime('%Y-%m-%d'),
                    'day_of_week': pred_date.strftime('%A'),
                    **result
                })
                
            except Exception as e:
                logger.error(f"Prediction failed for {pred_date}: {e}")
                # Fallback prediction
                predictions.append({
                    'date': pred_date.strftime('%Y-%m-%d'),
                    'day_of_week': pred_date.strftime('%A'),
                    'predicted_patients': 120,
                    'confidence_interval': [100, 140],
                    'model_confidence': 0.5
                })
        
        return predictions
    
    def save_model(self, filepath):
        """Save trained model and preprocessing pipeline"""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'model': self.model,
            'preprocessor': self.preprocessor,
            'feature_columns': self.feature_columns,
            'is_trained': self.is_trained,
            'timestamp': datetime.now().isoformat()
        }
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model and preprocessing pipeline"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.preprocessor = model_data['preprocessor']
        self.feature_columns = model_data.get('feature_columns', [])
        self.is_trained = model_data.get('is_trained', True)
        
        logger.info(f"Model loaded from {filepath}")
        return self

# Training script
if __name__ == "__main__":
    # Load data
    data_path = "data/hospital_data.csv"
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        logger.info("Please run generate_sample_data.py first")
        exit(1)
    
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} records from {data_path}")
    
    # Initialize and train predictor
    predictor = HospitalLoadPredictor()
    predictor, cv_results, feature_importance = predictor.train(df)
    
    # Save model
    model_path = "models/rf_model.joblib"
    predictor.save_model(model_path)
    
    # Test predictions
    logger.info("\nTesting predictions...")
    test_predictions = predictor.predict_next_7_days(
        city="Delhi",
        current_conditions={'aqi': 250, 'temperature': 24, 'outbreak': 0}
    )
    
    logger.info("7-Day Forecast for Delhi:")
    for pred in test_predictions:
        logger.info(f"{pred['date']} ({pred['day_of_week']}): {pred['predicted_patients']} patients "
                   f"(CI: {pred['confidence_interval'][0]}-{pred['confidence_interval'][1]})")
    
    logger.info("Model training completed successfully!")