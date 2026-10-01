# run_app.ps1 — Launch the Student Performance Prediction app
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Student Performance Prediction System" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Starting at http://localhost:8501 ..." -ForegroundColor Green
Write-Host "Press Ctrl+C to stop." -ForegroundColor Yellow
Write-Host ""
& "C:\Users\USER\AppData\Local\Python\bin\python.exe" -m streamlit run app.py --server.port 8501
