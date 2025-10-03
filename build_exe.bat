@echo off
set SCRIPT_DIR=%~dp0
set DIST_DIR=%SCRIPT_DIR%dist
if not exist "%DIST_DIR%" mkdir "%DIST_DIR%"
pyinstaller --onefile --clean --distpath "%DIST_DIR%" --name tcm-data-cleaner app.py
