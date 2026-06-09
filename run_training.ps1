# Glaucoma Detection - Enhanced Training Script (PowerShell)
# This script sets up the environment and runs enhanced training

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "Glaucoma Detection - Enhanced Training Setup" -ForegroundColor Cyan
Write-Host "===========================================================`n" -ForegroundColor Cyan

# Check if we're in the right directory
if (-Not (Test-Path "src\train.py")) {
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

# Install requirements
Write-Host "Installing requirements from requirements.txt..." -ForegroundColor Yellow
pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Error "Error: Failed to install requirements"
    Pause
    Exit 1
}

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "Starting Enhanced Training" -ForegroundColor Cyan
Write-Host "===========================================================`n" -ForegroundColor Cyan
Write-Host "Training configuration:" -ForegroundColor Green
Write-Host "  - Epochs: 40 (8 frozen backbone + 32 fine-tuning)" -ForegroundColor Green
Write-Host "  - Batch size: 8" -ForegroundColor Green
Write-Host "  - Enhanced augmentation: RandomResizedCrop, increased rotation, enhanced ColorJitter, RandomGrayscale, GaussianBlur" -ForegroundColor Green
Write-Host "  - Early stopping patience: 10 epochs" -ForegroundColor Green
Write-Host ""
Write-Host "The training will show progress below..." -ForegroundColor Green
Write-Host ""

# Start training
python src\train.py

# Deactivate virtual environment when done
Deactivate

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "Training completed!" -ForegroundColor Cyan
Write-Host "===========================================================`n" -ForegroundColor Cyan
Write-Host "To evaluate the trained model, run:" -ForegroundColor Yellow
Write-Host "   python src\evaluate.py" -ForegroundColor Yellow
Write-Host ""
Write-Host "The best model is saved at: outputs\models\best_model.pth" -ForegroundColor Yellow
Write-Host ""
Pause