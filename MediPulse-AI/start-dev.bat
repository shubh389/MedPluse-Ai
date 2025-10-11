@echo off
echo Starting MediPulse AI Development Environment...
echo.

echo [1/4] Checking dependencies...
cd /d "%~dp0"

if not exist "node_modules" (
    echo Installing root dependencies...
    npm install
)

if not exist "backend\node_modules" (
    echo Installing backend dependencies...
    cd backend
    npm install
    cd ..
)

if not exist "frontend\node_modules" (
    echo Installing frontend dependencies...
    cd frontend
    npm install
    cd ..
)

echo.
echo [2/4] Setting up ML Service...
cd ml-service
python -m pip install -r requirements.txt >nul 2>&1
cd ..

echo.
echo [3/4] Starting all services...
echo - ML Service: http://localhost:8000
echo - Backend API: http://localhost:5000
echo - Frontend: http://localhost:3000
echo.
echo Press Ctrl+C to stop all services
echo.

start /b cmd /c "cd ml-service && python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000"
timeout /t 3 /nobreak >nul

start /b cmd /c "cd backend && npm run dev"
timeout /t 3 /nobreak >nul

start /b cmd /c "cd frontend && npm start"

echo [4/4] All services started!
echo.
echo Opening dashboard in your browser...
timeout /t 5 /nobreak >nul
start http://localhost:3000

echo.
echo ===========================================
echo MediPulse AI is running!
echo Frontend: http://localhost:3000
echo Backend:  http://localhost:5000/api
echo ML API:   http://localhost:8000
echo ===========================================
echo.
pause