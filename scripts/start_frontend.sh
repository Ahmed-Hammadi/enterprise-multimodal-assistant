#!/bin/bash
# Start the Streamlit frontend

echo "Starting Enterprise Multimodal Assistant Frontend..."

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "Virtual environment not found. Please run start_backend.sh first."
    exit 1
fi

# Start the frontend
echo "Starting Streamlit UI on http://localhost:8501"
streamlit run frontend/streamlit_app.py
