@echo off
title AI-IDS Dashboard — COM668
echo.
echo  ======================================================
echo   AI-Based IDS Dashboard  ^|  COM668  ^|  Abdulbosit
echo  ======================================================
echo.
echo  Starting dashboard at http://localhost:8501
echo  Press Ctrl+C to stop.
echo.
cd /d "%~dp0"
python -m streamlit run dashboard.py --server.port 8501 --server.headless false --theme.base dark --theme.primaryColor "#3b82f6" --theme.backgroundColor "#060b14" --theme.secondaryBackgroundColor "#0d1525" --theme.textColor "#f1f5f9"
pause
