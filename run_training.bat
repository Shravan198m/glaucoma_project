@echo off
REM Glaucoma Detection - Enhanced Training Script
REM This script sets up the environment and runs enhanced training

echo.
echo ============================================================
echo Glaucoma Detection - Enhanced Training Setup
echo ============================================================
echo.

REM Check if we're in the right directory
if not exist "src\train.py" (
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

REM Install requirements
echo Installing requirements from requirements.txt...
pip install -r requirements.txt
if errorlevel 1 (
    echo Error: Failed to install requirements
    pause
    exit /b 1
)

echo.
echo ============================================================
echo Starting Enhanced Training
echo ============================================================
echo.
echo Training configuration:
echo  - Epochs: 40 (8 frozen backbone + 32 fine-tuning)
echo  - Batch size: 8
echo  - Enhanced augmentation: RandomResizedCrop, increased rotation, enhanced ColorJitter, RandomGrayscale, GaussianBlur
echo  - Early stopping patience: 10 epochs
echo.
echo The training will show progress below...
echo.

REM Start training
python src\train.py

REM Deactivate virtual environment when done
call venv\Scripts\deactivate

echo.
echo ============================================================
echo Training completed!
echo ============================================================
echo.
echo To evaluate the trained model, run:
echo   python src\evaluate.py
echo.
echo The best model is saved at: outputs\models\best_model.pth
echo.
pause