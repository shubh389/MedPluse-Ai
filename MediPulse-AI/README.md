# MediPulse AI - Hospital Patient Load Forecasting System

## Overview

MediPulse AI is an intelligent hospital management system that predicts patient load based on environmental factors (AQI, temperature), events (festivals), and health conditions (outbreaks). It provides actionable recommendations for staffing, medical supplies, and resource allocation.

## Architecture

```
Data Sources → ML Model (FastAPI) → Backend (Express.js) → Frontend (React)
     ↓              ↓                    ↓                    ↓
   CSV files → Random Forest → Resource Recommendations → Dashboard & Alerts
```

## Features

- **Patient Load Forecasting**: 7-day predictions using Random Forest
- **Resource Planning**: Automated staff and supply recommendations
- **Real-time Dashboard**: Interactive charts and monitoring
- **SMS Alerts**: Twilio integration for surge notifications
- **AI Explanations**: OpenAI-powered advisory generation
- **Continuous Learning**: Model retraining pipeline

## Quick Start

### Prerequisites

- Python 3.9+
- Node.js 16+
- npm or yarn

### Installation

1. **Clone and setup**

```bash
git clone <repository>
cd MediPulse-AI
```

2. **Start ML Service (FastAPI)**

```bash
cd ml-service
pip install -r requirements.txt
python generate_sample_data.py  # Generate test data
python train_model.py           # Train initial model
uvicorn main:app --reload --port 8000
```

3. **Start Backend (Express.js)**

```bash
cd backend
npm install
cp .env.example .env  # Configure environment variables
npm run dev  # Starts on port 3001
```

4. **Start Frontend (React)**

```bash
cd frontend
npm install
npm start  # Starts on port 3000
```

### Environment Variables

Create `.env` files in both `backend/` and `ml-service/`:

**Backend (.env)**

```env
PORT=3001
FASTAPI_URL=http://localhost:8000
TWILIO_ACCOUNT_SID=your_twilio_sid
TWILIO_AUTH_TOKEN=your_twilio_token
TWILIO_PHONE_NUMBER=+1234567890
OPENAI_API_KEY=your_openai_key
```

**ML Service (.env)**

```env
PORT=8000
MODEL_PATH=./models/rf_model.joblib
DATA_PATH=./data/hospital_data.csv
```

## API Endpoints

### ML Service (Port 8000)

- `POST /predict` - Single patient prediction
- `POST /predict/batch` - Batch predictions
- `GET /health` - Service health check

### Backend (Port 3001)

- `POST /api/predict` - Enhanced prediction with recommendations
- `POST /api/alerts/send` - Send SMS alerts
- `GET /api/health` - Backend health status

### Frontend (Port 3000)

- Dashboard with forecasting charts
- Prediction input form
- Resource recommendations display
- Alert management interface

## Development

### Project Structure

```
MediPulse-AI/
├── ml-service/          # Python FastAPI service
│   ├── models/          # Trained ML models
│   ├── data/           # Training data
│   └── main.py         # FastAPI app
├── backend/            # Node.js Express service
│   ├── routes/         # API routes
│   ├── services/       # Business logic
│   └── server.js       # Express app
├── frontend/           # React dashboard
│   ├── src/components/ # React components
│   ├── src/services/   # API clients
│   └── public/         # Static assets
└── data/              # Shared data files
```

### Testing

```bash
# Test ML service
cd ml-service && python -m pytest tests/

# Test backend
cd backend && npm test

# Test frontend
cd frontend && npm test
```

### Docker Deployment

```bash
docker-compose up --build
```

## Usage Examples

### 1. Make a Prediction

```bash
curl -X POST "http://localhost:3001/api/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "city": "Delhi",
    "date": "2025-11-01",
    "aqi": 320,
    "temperature": 24.5,
    "festival": "Diwali",
    "outbreak": 1
  }'
```

### 2. Dashboard Access

Navigate to `http://localhost:3000` to access the interactive dashboard.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

For issues and questions, please open a GitHub issue or contact the development team.
