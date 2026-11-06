@echo off
title RUAS-SIPIL AI Dashboard Launcher
echo Memulai Sistem RUAS-SIPIL...
cd /d "%~dp0"
call venv\Scripts\activate.bat
start http://127.0.0.1:7860
python app.py
pause