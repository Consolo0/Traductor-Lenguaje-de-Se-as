@echo off
cd /d "%~dp0"

echo === Instalando dependencias en el venv ===
venv\Scripts\python.exe -m pip install -r requirements.txt

echo.
echo === Extrayendo landmarks ===
venv\Scripts\python.exe models\LandmarkExtractor.py --input "data\asl_dataset" --output "data\landmarks\landmarks.csv"

echo.
echo === Listo ===
pause