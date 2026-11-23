@echo off
rem Runs Wadia with a console that shows what it's doing (useful for troubleshooting).
rem For everyday use, open the Wadia shortcut on the Desktop or in the Start menu.
title Wadia
cd /d "%~dp0"

rem Settings: remove "rem " at the start of a line to use it.
rem set MIC_DEVICE=Intel
rem set PHONE=off

".venv\Scripts\python.exe" local\app.py
echo.
pause
