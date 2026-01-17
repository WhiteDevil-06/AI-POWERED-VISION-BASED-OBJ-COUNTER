@echo off
setlocal
title AI VISION DASHBOARD - INSTALLER & LAUNCHER

:: 1. CHECK PYTHON
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH!
    echo Please install Python 3.10+ and tick "Add to PATH"
    pause
    exit /b
)

:: 2. CHECK VIRTUAL ENV
if not exist "env\" (
    echo [INFO] First time setup detected...
    echo [INFO] Creating Virtual Environment 'env'...
    python -m venv env
    
    echo [INFO] Activating Environment...
    call .\env\Scripts\activate.bat
    
    echo [INFO] Upgrading PIP...
    python -m pip install --upgrade pip
    
    echo [INFO] Installing Dependencies (This may take a minute)...
    pip install -r requirements.txt
    
    echo [SUCCESS] Setup Complete!
) else (
    echo [INFO] Virtual Environment found.
    call .\env\Scripts\activate.bat
)

:: 3. LAUNCH APP
echo.
echo =================================================
echo    🚀 STARTING AI VISION SYSTEM...
echo    (Press Ctrl+C to Stop)
echo =================================================
echo.

streamlit run dashboard.py

pause
