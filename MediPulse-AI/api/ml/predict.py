import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import os

# Cache for model and encoders
_model = None
_encoders = {}

def load_model():
    """Load model and encoders, creating them if they don't exist"""
    global _model, _encoders
    
    if _model is None:
        try:
            # Try to load existing model (this would be uploaded as part of deployment)
            # For now, create a simple model
            _model = RandomForestRegressor(
                n_estimators=50,
                max_depth=10,
                random_state=42
            )
            
            # Create label encoders for categorical features
            categorical_features = ['season', 'weather_conditions', 'local_events', 'city']
            for feature in categorical_features:
                _encoders[feature] = LabelEncoder()
            
            # Train with sample data for demonstration
            sample_data = create_sample_training_data()
            X, y = prepare_features(sample_data)
            _model.fit(X, y)
            
        except Exception as e:
            print(f"Error loading model: {e}")
            # Create a fallback simple model
            _model = RandomForestRegressor(n_estimators=10, random_state=42)
            
    return _model, _encoders

def create_sample_training_data():
    """Create sample training data for the model"""
    np.random.seed(42)
    
    data = []
    cities = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata']
    seasons = ['spring', 'summer', 'monsoon', 'winter']
    weather_conditions = ['clear', 'cloudy', 'rainy', 'stormy', 'foggy']
    local_events = ['none', 'festival', 'sports_event', 'concert', 'political_rally']
    
    for _ in range(1000):
        city = np.random.choice(cities)
        season = np.random.choice(seasons)
        weather = np.random.choice(weather_conditions)
        event = np.random.choice(local_events)
        
        # Base patient load
        base_load = np.random.normal(100, 20)
        
        # Seasonal adjustments
        seasonal_multiplier = {
            'summer': 1.2, 'monsoon': 1.4, 'winter': 1.1, 'spring': 1.0
        }
        base_load *= seasonal_multiplier.get(season, 1.0)
        
        # Weather adjustments
        weather_multiplier = {
            'stormy': 1.3, 'rainy': 1.2, 'foggy': 1.15, 'cloudy': 1.05, 'clear': 1.0
        }
        base_load *= weather_multiplier.get(weather, 1.0)
        
        # Event adjustments
        event_multiplier = {
            'festival': 1.5, 'sports_event': 1.2, 'concert': 1.1, 'political_rally': 1.15, 'none': 1.0
        }
        base_load *= event_multiplier.get(event, 1.0)
        
        # AQI impact
        aqi = np.random.randint(20, 300)
        if aqi > 200:
            base_load *= 1.25
        elif aqi > 100:
            base_load *= 1.1
        
        # Day of week impact
        day_of_week = np.random.randint(0, 7)
        if day_of_week in [5, 6]:  # Weekend
            base_load *= 0.9
        
        data.append({
            'day_of_week': day_of_week,
            'month': np.random.randint(1, 13),
            'season': season,
            'weather_conditions': weather,
            'holiday': np.random.choice([True, False], p=[0.1, 0.9]),
            'local_events': event,
            'air_quality_index': aqi,
            'temperature': np.random.randint(15, 40),
            'humidity': np.random.randint(30, 90),
            'city': city,
            'patient_count': max(int(base_load), 20)
        })
    
    return pd.DataFrame(data)

def prepare_features(data):
    """Prepare features for model training/prediction"""
    global _encoders
    
    df = data.copy()
    
    # Encode categorical features
    categorical_features = ['season', 'weather_conditions', 'local_events', 'city']
    
    for feature in categorical_features:
        if feature in df.columns:
            if feature in _encoders:
                # For prediction, handle unseen categories
                try:
                    df[feature] = _encoders[feature].transform(df[feature])
                except ValueError:
                    # Handle unseen categories by assigning a default value
                    known_categories = _encoders[feature].classes_
                    df[feature] = df[feature].apply(
                        lambda x: x if x in known_categories else known_categories[0]
                    )
                    df[feature] = _encoders[feature].transform(df[feature])
            else:
                # For training, fit the encoder
                df[feature] = _encoders[feature].fit_transform(df[feature])
    
    # Select features for prediction
    feature_columns = [
        'day_of_week', 'month', 'season', 'weather_conditions',
        'holiday', 'local_events', 'air_quality_index', 'temperature',
        'humidity', 'city'
    ]
    
    X = df[feature_columns]
    
    # Convert boolean to int
    if 'holiday' in X.columns:
        X['holiday'] = X['holiday'].astype(int)
    
    if 'patient_count' in df.columns:
        y = df['patient_count']
        return X, y
    else:
        return X, None

def handler(request):
    """Main handler function for Vercel"""
    try:
        # Parse request body
        if hasattr(request, 'get_json'):
            data = request.get_json()
        else:
            # Handle different request formats
            body = request.body if hasattr(request, 'body') else request
            if isinstance(body, bytes):
                body = body.decode('utf-8')
            if isinstance(body, str):
                data = json.loads(body)
            else:
                data = body
        
        # Load model
        model, encoders = load_model()
        
        # Validate input
        required_fields = ['day_of_week', 'month', 'season', 'weather_conditions', 'city']
        for field in required_fields:
            if field not in data:
                return {
                    'statusCode': 400,
                    'body': json.dumps({'error': f'Missing required field: {field}'})
                }
        
        # Set default values for optional fields
        data.setdefault('holiday', False)
        data.setdefault('local_events', 'none')
        data.setdefault('air_quality_index', 50)
        data.setdefault('temperature', 25)
        data.setdefault('humidity', 60)
        
        # Create DataFrame from input
        input_df = pd.DataFrame([data])
        
        # Prepare features
        X, _ = prepare_features(input_df)
        
        # Make prediction
        prediction = model.predict(X)[0]
        
        # Calculate confidence score based on feature values
        confidence_score = 0.85
        if data.get('air_quality_index', 0) > 200:
            confidence_score *= 0.9
        if data.get('local_events') != 'none':
            confidence_score *= 0.95
        
        # Generate confidence interval
        std_dev = prediction * 0.15
        confidence_interval = [
            max(int(prediction - std_dev), 0),
            int(prediction + std_dev)
        ]
        
        result = {
            'predicted_patients': int(max(prediction, 0)),
            'confidence_score': round(confidence_score, 3),
            'confidence_interval': confidence_interval,
            'model_version': '1.0.0',
            'timestamp': datetime.now().isoformat()
        }
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'POST, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type'
            },
            'body': json.dumps(result)
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'error': 'Internal server error',
                'message': str(e)
            })
        }

# Vercel serverless function entry point
def lambda_handler(event, context):
    """AWS Lambda compatible handler"""
    return handler(event)

# For local testing
if __name__ == "__main__":
    # Test the function
    test_data = {
        'day_of_week': 1,
        'month': 12,
        'season': 'winter',
        'weather_conditions': 'clear',
        'city': 'Mumbai',
        'air_quality_index': 75
    }
    
    class MockRequest:
        def __init__(self, data):
            self.body = json.dumps(data)
    
    result = handler(MockRequest(test_data))
    print(json.dumps(result, indent=2))