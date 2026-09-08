@echo off
title Ejecutando Inferencia en Tiempo Real
cd /d "%~dp0"

set PYTHON_PATH="c:\Users\conso\AppData\Local\Programs\Python\Python312\python.exe"

%PYTHON_PATH% models/realtime_infer.py --model "models/mlp_model.pt" --classes "models/mlp_model.classes.json" --min-detection-confidence 0.5 --min-model-confidence 0.6 --hold-seconds 0.8 --space-gap-seconds 1.5 --camera-index 0

echo.
echo Proceso finalizado.
pause
