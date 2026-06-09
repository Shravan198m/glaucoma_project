@echo off
echo ============================================================
echo Fixing Windows Long Path Support for Glaucoma Project
echo ============================================================
echo.
echo This script will help enable Windows Long Path support,
echo which is required for installing Python packages with long paths.
echo.
echo IMPORTANT: You need to RUN THIS AS ADMINISTRATOR
echo.
echo Choose an option:
echo 1. Enable Long Paths via Registry (works on all Windows versions)
echo 2. Show manual instructions for Group Policy
echo 3. Exit
echo.
set /p choice="Enter choice (1-3): "

if "%choice%"=="1" (
    echo.
    echo Enabling Long Paths via Registry...
    reg add "HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem" /v LongPathsEnabled /t REG_DWORD /d 1 /f
    if %errorlevel% equ 0 (
        echo.
        echo SUCCESS: Long Paths enabled via registry.
        echo YOU MUST RESTART YOUR COMPUTER for this to take effect.
    ) else (
        echo.
        echo ERROR: Failed to modify registry. Please run this script as Administrator.
    )
) else if "%choice%"=="2" (
    echo.
    echo ============================================================
    echo MANUAL INSTRUCTIONS - Group Policy Method
    echo ============================================================
    echo.
    echo 1. Press Win + R, type: gpedit.msc, press Enter
    echo 2. Navigate to:
    echo    Computer Configuration -> Administrative Templates -> System -> Filesystem
    echo 3. Double-click: "Enable Win32 long paths"
    echo 4. Select: Enabled
    echo 5. Click OK
    echo 6. RESTART YOUR COMPUTER
    echo.
    echo NOTE: gpedit.msc is only available on Windows Pro/Education/Enterprise.
    echo If you have Windows Home, use option 1 (Registry method) above.
) else if "%choice%"=="3" (
    echo.
    echo Exiting without changes.
) else (
    echo.
    
    echo Invalid choice. Please run again and select 1, 2, or 3.
)

echo.
pause