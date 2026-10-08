@echo off
cd /d "%~dp0backend"

echo ============================================
echo    VetAI - starting up
echo ============================================
echo.

REM --- 1. Check Python exists ---
python --version >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Python was not found on this computer.
  echo.
  echo Install Python 3.10 - 3.13 from:  https://www.python.org/downloads/
  echo During install, TICK the box "Add Python to PATH".
  goto stop
)
for /f "delims=" %%v in ('python --version') do echo Using %%v
echo.

REM --- 2. Create the virtual environment if missing ---
if not exist ".venv\Scripts\activate.bat" (
  echo Creating environment...
  python -m venv .venv
  if errorlevel 1 ( echo [ERROR] Could not create the environment. & goto stop )
)
call .venv\Scripts\activate.bat

REM --- 3. Core dependencies (required) ---
echo Installing core dependencies ^(first run takes a few minutes^)...
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt
if errorlevel 1 ( echo [ERROR] Could not install core dependencies. Check your internet connection. & goto stop )

REM --- 4. Photo dependencies (OPTIONAL - failure is NOT fatal) ---
if exist "app\ml\image_model.keras" (
  echo Installing photo-detection support ^(TensorFlow - large, please wait^)...
  python -m pip install --quiet -r requirements-image.txt
  if errorlevel 1 (
    echo.
    echo [NOTE] TensorFlow did not install - that is OK.
    echo VetAI will run with SYMPTOMS ONLY. You can add photo support later.
    echo.
  )
)

REM --- 5. Train the symptom model if needed ---
if not exist "app\ml\symptom_model.joblib" (
  echo Preparing the disease model ^(one-time, about a minute^)...
  python train_symptom_model.py
  if errorlevel 1 ( echo [ERROR] Model preparation failed. & goto stop )
)

echo.
echo ============================================
echo    VetAI is running.
echo    Open this address in your browser:
echo.
echo        http://127.0.0.1:8000
echo.
echo    Keep this window OPEN while using VetAI.
echo    Press CTRL+C here to stop it.
echo.
echo Launching your web browser...
start "" http://127.0.0.1:8000
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

echo.
echo VetAI has stopped.

:stop
echo.
echo -------------------------------------------------------
echo This window will stay open so you can read any messages.
echo Press any key to close it.
pause >nul
