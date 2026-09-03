@echo off
cd /d "%~dp0"

set PYTHON_EXE=c:\Users\conso\AppData\Local\Programs\Python\Python312\python.exe

echo === Instalando dependencias ===
"%PYTHON_EXE%" -m pip install -r requirements.txt

echo.
echo === Extrayendo landmarks ===
"%PYTHON_EXE%" models\LandmarkExtractor.py --input "data\asl_dataset" --output "data\landmarks\landmarks.csv"

echo.
echo === Listo ===
pause