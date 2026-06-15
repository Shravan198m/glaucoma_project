@echo off
REM Glaucoma Detection API Server
REM This script sets up the environment and runs the API server

echo.
echo ============================================================
echo Glaucoma Detection - API Server Setup
echo ============================================================
echo.

REM Check if we're in the right directory
if not exist "src\api.py" (
    echo Error: Please run this script from the glaucoma_project directory
    echo Current directory: %cd%
    echo.
    pause
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo Error: Failed to create virtual environment
        pause
        exit /b 1
    )
) else (
    echo Virtual environment already exists.
)

REM Activate virtual environment and install requirements
echo.
echo Activating virtual environment and installing dependencies...
call venv\Scripts\activate
if errorlevel 1 (
    echo Error: Failed to activate virtual environment
    pause
    exit /b 1
)

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo Warning: Failed to upgrade pip (continuing anyway)
)

REM Install requirements (including API dependencies)
echo Installing requirements from requirements.txt...
pip install -r requirements.txt
if errorlevel 1 (
    echo Error: Failed to install requirements
    pause
    exit /b 1
)

echo.
echo ============================================================
echo Starting API Server
echo ============================================================
echo.
echo The API will be available at:
echo   http://localhost:8000
echo   http://127.0.0.1:8000
echo.
echo API documentation (Swagger UI) will be available at:
echo   http://localhost:8000/docs
echo   http://127.0.0.1:8000/redoc
echo.
echo Press CTRL+C to stop the server
echo.

REM Start the API server
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

REM Deactivate virtual environment when done
call venv\Scripts\deactivate

echo.
echo ============================================================
echo API Server stopped!
echo ============================================================
echo.
pause