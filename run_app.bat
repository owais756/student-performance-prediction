@echo off
echo ============================================
echo   Student Performance Prediction System
echo ============================================
echo.
echo Starting Streamlit app on http://localhost:8501
echo Press Ctrl+C to stop.
echo.
"C:\Users\USER\AppData\Local\Python\bin\python.exe" -m streamlit run app.py --server.port 8501
pause
