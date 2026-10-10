@echo off
cd /d "%~dp0"
python run.py
if errorlevel 1 (
  echo.
  echo Orbit Solver skoncil s chybou. Podrobnosti jsou take v orbit-crash.log.
  pause
)
