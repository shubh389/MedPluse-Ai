#!/bin/bash

echo "🚀 Starting MediPulse AI Development Environment..."
echo

# Get the script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$DIR"

echo "📦 [1/4] Checking dependencies..."

# Install root dependencies
if [ ! -d "node_modules" ]; then
    echo "Installing root dependencies..."
    npm install
fi

# Install backend dependencies
if [ ! -d "backend/node_modules" ]; then
    echo "Installing backend dependencies..."
    cd backend && npm install && cd ..
fi

# Install frontend dependencies
if [ ! -d "frontend/node_modules" ]; then
    echo "Installing frontend dependencies..."
    cd frontend && npm install && cd ..
fi

echo
echo "🤖 [2/4] Setting up ML Service..."
cd ml-service
pip install -r requirements.txt > /dev/null 2>&1
cd ..

echo
echo "⚡ [3/4] Starting all services..."
echo "- ML Service: http://localhost:8000"
echo "- Backend API: http://localhost:5000"
echo "- Frontend: http://localhost:3000"
echo
echo "Press Ctrl+C to stop all services"
echo

# Start ML Service
echo "Starting ML Service..."
cd ml-service
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000 &
ML_PID=$!
cd ..
sleep 3

# Start Backend
echo "Starting Backend API..."
cd backend
npm run dev &
BACKEND_PID=$!
cd ..
sleep 3

# Start Frontend
echo "Starting Frontend..."
cd frontend
npm start &
FRONTEND_PID=$!
cd ..
sleep 5

echo
echo "🎉 [4/4] All services started!"
echo
echo "==========================================="
echo "🏥 MediPulse AI is running!"
echo "Frontend: http://localhost:3000"
echo "Backend:  http://localhost:5000/api"
echo "ML API:   http://localhost:8000"
echo "==========================================="
echo
echo "Opening dashboard in your browser..."

# Try to open browser (works on most systems)
if command -v xdg-open > /dev/null; then
    xdg-open http://localhost:3000
elif command -v open > /dev/null; then
    open http://localhost:3000
elif command -v start > /dev/null; then
    start http://localhost:3000
fi

# Wait for Ctrl+C
trap "echo; echo 'Stopping all services...'; kill $ML_PID $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT
wait