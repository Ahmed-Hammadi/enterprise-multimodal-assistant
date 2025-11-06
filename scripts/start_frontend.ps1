# Start the Streamlit frontend
Write-Host "Starting Enterprise Multimodal Assistant Frontend..." -ForegroundColor Green

# Check if virtual environment exists
if (Test-Path "venv\Scripts\Activate.ps1") {
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    & venv\Scripts\Activate.ps1
} else {
    Write-Host "Virtual environment not found. Please run start_backend.ps1 first." -ForegroundColor Red
    exit 1
}

# Start the frontend
Write-Host "Starting Streamlit UI on http://localhost:8501" -ForegroundColor Green
streamlit run frontend\streamlit_app.py
