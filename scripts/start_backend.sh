#!/bin/bash
# Start the FastAPI backend server

echo "Starting Enterprise Multimodal Assistant Backend..."

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "Virtual environment not found. Creating one..."
    python3 -m venv venv
    source venv/bin/activate
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Start the backend
echo "Starting backend server on http://localhost:8000"
echo "API docs available at http://localhost:8000/docs"
cd backend
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
