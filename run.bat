@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\activate.bat (
    echo Run setup_windows.bat first.
    pause
    exit /b 1
)
call .venv\Scripts\activate.bat
set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
echo Starting Vidya Offline... (close this window to stop)
streamlit run app.py
pause
