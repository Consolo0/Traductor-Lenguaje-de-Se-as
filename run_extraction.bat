@echo off
REM Se ejecuta desde la raíz del proyecto (donde está requirements.txt)
REM Doble click o correr desde cmd/PowerShell: run_extraction.bat

cd /d "%~dp0"

echo === Instalando dependencias ===
pip install -r requirements.txt

echo.
echo === Extrayend
python models\LandmarkExtractor.py --input landmarks === "data\asl_dataset" --output "data\landmarks\landmarks.csv"

echo.
echo === Listo ===
pause