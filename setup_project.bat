@echo off
setlocal
cd /d "%~dp0"

echo ===================================================
echo   BackTrans-Metrics-Hub - Setup
echo ===================================================

echo This script will set up the environment and install dependencies.
echo.

:: Check for existing venv
if not exist "venv\" goto CREATE_VENV

:ASK_CLEAN
echo [!] Found existing virtual environment (venv).
set /p choice="Do you want to perform a CLEAN install? (y/n): "
if /i "%choice%"=="y" goto DO_CLEAN
echo [1/3] Skipping venv creation (already exists).
goto INSTALL_DEPS

:DO_CLEAN
echo [1/3] Deleting existing venv...
rmdir /s /q "venv"
if exist "venv" (
    echo [!] Error: Could not delete venv folder. Please close any programs using it.
    pause
    exit /b 1
)

:CREATE_VENV
echo [1/3] Creating Virtual Environment...
python -m venv venv
if errorlevel 1 (
    echo [!] Error: Failed to create virtual environment. 
    echo Make sure Python is installed and in your PATH.
    pause
    exit /b 1
)

:INSTALL_DEPS
echo [2/3] Activating Environment and Installing Dependencies...
echo This will install GPU-enabled PyTorch (CUDA 12.1) and all evaluation metrics.
echo.

"venv\Scripts\python.exe" -m pip install --upgrade pip
"venv\Scripts\pip.exe" install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [!] Error: Installation failed.
    pause
    exit /b 1
)

:: Check for .env file
if exist ".env" goto SETUP_DONE
echo.
echo [!] WARNING: .env file not found! 
echo Please copy .env.example to .env and fill in your API keys.

:SETUP_DONE
echo.
echo [3/3] Setup Complete!
echo.
echo ===================================================
echo   To run the server:
echo   venv\Scripts\uvicorn app.main:app --reload
echo ===================================================
echo.
pause
endlocal
