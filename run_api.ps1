# Glaucoma Detection API Server (PowerShell)
# This script sets up the environment and runs the API server

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "Glaucoma Detection - API Server Setup" -ForegroundColor Cyan
Write-Host "===========================================================`n" -ForegroundColor Cyan

# Check if we're in the right directory
if (-Not (Test-Path "src\api.py")) {
    Write-Error "Error: Please run this script from the glaucoma_project directory"
    Write-Error "Current directory: $pwd"
    Pause
    Exit 1
}

# Create virtual environment if it doesn't exist
if (-Not (Test-Path "venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Error: Failed to create virtual environment"
        Pause
        Exit 1
    }
} else {
    Write-Host "Virtual environment already exists." -ForegroundColor Green
}

# Activate virtual environment and install requirements
Write-Host "`nActivating virtual environment and installing dependencies..." -ForegroundColor Yellow
& venv\Scripts\Activate.ps1
if ($LASTEXITCODE -ne 0) {
    Write-Error "Error: Failed to activate virtual environment"
    Pause
    Exit 1
}

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Warning: Failed to upgrade pip (continuing anyway)"
}

# Install requirements (including API dependencies)
Write-Host "Installing requirements from requirements.txt..." -ForegroundColor Yellow
pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Error "Error: Failed to install requirements"
    Pause
    Exit 1
}

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "Starting API Server" -ForegroundColor Cyan
Write-Host "===========================================================`n" -ForegroundColor Cyan
Write-Host "The API will be available at:" -ForegroundColor Green
Write-Host "   http://localhost:8000" -ForegroundColor Green
Write-Host "   http://127.0.0.1:8000" -ForegroundColor Green
Write-Host ""
Write-Host "API documentation (Swagger UI) will be available at:" -ForegroundColor Green
Write-Host "   http://localhost:8000/docs" -ForegroundColor Green
Write-Host "   http://127.0.0.1:8000/redoc" -ForegroundColor Green
Write-Host ""
Write-Host "Press CTRL+C to stop the server" -ForegroundColor Yellow
Write-Host ""

# Start the API server
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

# Deactivate virtual environment when done
Deactivate

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "API Server stopped!" -ForegroundColor Cyan
Write-Host "===========================================================`n" -ForegroundColor Cyan
Write-Host ""
Pause