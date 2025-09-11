@echo off
REM Windows helper to rebuild the results.html dashboard locally
python tmr.py dash
exit /b %errorlevel%

