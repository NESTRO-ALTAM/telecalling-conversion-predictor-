@echo off
REM Double-click this file to launch the Telecalling Conversion Predictor.
cd /d "%~dp0"
if not exist model_artifacts.pkl (
    echo Model not found - generating data and training...
    venv\Scripts\python.exe generate_data.py
    venv\Scripts\python.exe train.py
)
echo Starting dashboard at http://localhost:8501  (close this window to stop)
venv\Scripts\streamlit.exe run app.py
pause
