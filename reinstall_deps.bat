@echo off
echo ============================================================
echo Reinstalling Dependencies for Glaucoma Project
echo ============================================================
echo.
echo This script will:
echo 1. Remove the incomplete virtual environment
echo 2. Create a fresh virtual environment
echo 3. Install all dependencies
echo 4. Start the training
echo.
echo MAKE SURE YOU HAVE ENABLED WINDOWS LONG PATHS FIRST!
echo (Run fix_long_paths.bat as Administrator and restart)
echo.
pause
echo.
echo Step 1: Removing incomplete virtual environment...
cd /d "C:\Users\svmoo\OneDrive\Documents\GLUCOMA\glaucoma_project"
rmdir /s /q venv
if %errorlevel% neq 0 (
    echo Warning: Could not remove venv (might not exist or be in use)
)
echo.
echo Step 2: Creating fresh virtual environment...
python -m venv venv
if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)
echo.
echo Step 3: Activating virtual environment...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)
echo.
echo Step 4: Upgrading pip...
python -m pip install --upgrade pip
if %errorlevel% neq 0 (
    echo WARNING: Failed to upgrade pip (continuing anyway)
)
echo.
echo Step 5: Installing requirements...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install requirements
    echo This likely means Long Paths are still not enabled properly
    pause
    exit /b 1
)
echo.
echo Step 6: Starting training...
echo.
python src\train.py
echo.
echo Training completed or interrupted.
echo.
call venv\Scripts\deactivate.bat
echo.
pause