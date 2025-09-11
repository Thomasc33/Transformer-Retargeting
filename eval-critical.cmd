@echo off
REM Windows helper to run the critical experiment set and refresh dashboard
python tmr.py eval --set critical
exit /b %errorlevel%

