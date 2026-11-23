@echo off
setlocal
cd /d "%~dp0"
title Vidya Offline - setup
echo ============================================
echo   Vidya Offline - one-time setup (Windows)
echo ============================================
echo.

where python >nul 2>nul
if errorlevel 1 goto nopython

python -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)"
if errorlevel 1 echo [WARN] Python 3.11 or 3.12 is recommended. Continuing anyway...

if not exist .venv\Scripts\python.exe (
    echo Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 goto fail
)
call .venv\Scripts\activate.bat

echo Installing Python packages (needs internet, ~500 MB)...
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 goto fail

echo.
choice /c YN /m "Install OFFLINE TRANSLATION for Hindi, Kannada, Tamil etc. (about 3 GB download, needs 8 GB RAM to convert)"
if errorlevel 2 goto skiptr
pip install -r requirements-translation.txt
if errorlevel 1 goto fail
python scripts\download_translation_model.py --slim
:skiptr

echo.
where ollama >nul 2>nul
if errorlevel 1 goto noollama
echo Downloading AI models with Ollama...
ollama pull nomic-embed-text
ollama pull llama3.2
echo.
echo (On a 4 GB PC use the smaller chat model as well:  ollama pull llama3.2:1b)
goto verify

:noollama
echo [!] Ollama is not installed. Install it from https://ollama.com/download , start it, then run:
echo       ollama pull nomic-embed-text
echo       ollama pull llama3.2

:verify
echo.
python scripts\check_setup.py
echo.
echo Setup finished. Start the app by double-clicking run.bat
pause
exit /b 0

:nopython
echo [FAIL] Python was not found. Install Python 3.11 or 3.12 from https://www.python.org/downloads/
echo        and tick "Add python.exe to PATH" in the installer. Then run this file again.
pause
exit /b 1

:fail
echo.
echo [FAIL] Something went wrong - read the messages above.
pause
exit /b 1
