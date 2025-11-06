# Setup Guide for Enterprise Multimodal Assistant

This guide will walk you through setting up the Enterprise Multimodal Assistant on your local machine.

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Installation Steps](#installation-steps)
3. [LLM Configuration](#llm-configuration)
4. [First Run](#first-run)
5. [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 20.04+)
- **Python**: 3.8 or higher
- **RAM**: 4GB (8GB+ recommended for better performance)
- **Disk Space**: 5GB free space (for models and documents)
- **Internet**: Required for initial setup only

### Recommended Requirements
- **RAM**: 16GB+ (for running local LLMs smoothly)
- **GPU**: NVIDIA GPU with 6GB+ VRAM (optional, for faster embeddings)
- **CPU**: Multi-core processor (4+ cores)

## Installation Steps

### Step 1: Install Python

#### Windows
1. Download Python from [python.org](https://www.python.org/downloads/)
2. Run the installer and **check "Add Python to PATH"**
3. Verify installation:
   ```powershell
   python --version
   ```

#### macOS
```bash
# Using Homebrew
brew install python@3.11
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

### Step 2: Clone the Repository

```bash
git clone https://github.com/Ahmed-Hammadi/enterprise-multimodal-assistant.git
cd enterprise-multimodal-assistant
```

### Step 3: Create Virtual Environment

#### Windows (PowerShell)
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

#### macOS/Linux
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 4: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install:
- FastAPI & Uvicorn (backend server)
- Streamlit (frontend UI)
- Sentence Transformers (embeddings)
- FAISS (vector database)
- PyMuPDF, python-docx, pandas (document processing)
- OpenAI SDK (LLM integration)

**Note**: First installation may take 5-10 minutes as it downloads ML models.

### Step 5: Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your preferred text editor
# Windows: notepad .env
# macOS/Linux: nano .env
```

## LLM Configuration

You have three options for the LLM backend:

### Option A: Ollama (Recommended for Privacy)

**Advantages**: Completely local, free, easy to use

1. **Install Ollama**
   - Windows/macOS: Download from [ollama.ai](https://ollama.ai/)
   - Linux:
     ```bash
     curl -fsSL https://ollama.ai/install.sh | sh
     ```

2. **Download a model**
   ```bash
   # Recommended models (choose one):
   ollama pull llama2        # 7B parameters, good balance
   ollama pull mistral       # 7B parameters, fast and accurate
   ollama pull llama3        # 8B parameters, more capable
   ```

3. **Verify Ollama is running**
   ```bash
   ollama list  # Should show your downloaded models
   ```

4. **Configure in .env**
   ```env
   LLM_PROVIDER=ollama
   LLM_MODEL=llama2
   LLM_BASE_URL=http://localhost:11434
   ```

### Option B: LM Studio

**Advantages**: User-friendly GUI, local execution

1. **Download LM Studio** from [lmstudio.ai](https://lmstudio.ai/)
2. **Download a model** through LM Studio's interface:
   - Search for "Mistral" or "Llama"
   - Recommended: TheBloke's quantized models (GGUF format)
3. **Start the server**:
   - In LM Studio, go to "Local Server"
   - Click "Start Server"
   - Note the port (usually 1234)

4. **Configure in .env**
   ```env
   LLM_PROVIDER=lmstudio
   LLM_MODEL=mistral-7b-instruct
   LLM_BASE_URL=http://localhost:1234/v1
   ```

### Option C: OpenAI API

**Advantages**: Most capable, no local resources needed

1. **Get API Key** from [platform.openai.com](https://platform.openai.com/)
2. **Configure in .env**
   ```env
   LLM_PROVIDER=openai
   LLM_MODEL=gpt-3.5-turbo
   OPENAI_API_KEY=sk-your-api-key-here
   ```

**Note**: This option sends data to OpenAI servers and incurs API costs.

## First Run

### Step 1: Start the Backend

#### Windows
```powershell
# Option 1: Using helper script
.\scripts\start_backend.ps1

# Option 2: Manual
cd backend
python main.py
```

#### macOS/Linux
```bash
# Option 1: Using helper script
chmod +x scripts/start_backend.sh
./scripts/start_backend.sh

# Option 2: Manual
cd backend
python main.py
```

**Expected Output**:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 2: Verify Backend

Open your browser and visit:
- **Health Check**: http://localhost:8000/health
- **API Docs**: http://localhost:8000/docs

You should see the API documentation and status information.

### Step 3: Start the Frontend

**Open a new terminal** (keep backend running) and:

#### Windows
```powershell
# Option 1: Using helper script
.\scripts\start_frontend.ps1

# Option 2: Manual
streamlit run frontend\streamlit_app.py
```

#### macOS/Linux
```bash
# Option 1: Using helper script
chmod +x scripts/start_frontend.sh
./scripts/start_frontend.sh

# Option 2: Manual
streamlit run frontend/streamlit_app.py
```

**Expected Output**:
```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

The browser should automatically open the application.

### Step 4: Test the Application

1. **Upload a Document**:
   - Use the sidebar to upload a test document (PDF, Word, Excel, etc.)
   - Click "Upload & Process"
   - Wait for processing to complete

2. **Ask a Question**:
   - Type a question about the document in the chat input
   - Press "Send"
   - Review the response and sources

## Troubleshooting

### Backend Issues

#### "Port 8000 is already in use"
```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <process_id> /F

# macOS/Linux
lsof -ti:8000 | xargs kill -9
```

#### "No module named 'fastapi'"
```bash
# Activate virtual environment first
pip install -r requirements.txt
```

#### "ModuleNotFoundError: No module named 'rag_pipeline'"
```bash
# Make sure you're in the correct directory
cd backend
python main.py
```

### Frontend Issues

#### "Streamlit not found"
```bash
pip install streamlit
```

#### "Backend server is not running"
- Ensure backend is started first (Step 1)
- Check if backend is accessible at http://localhost:8000/health

### LLM Issues

#### Ollama: "Connection refused"
```bash
# Start Ollama service
# Windows/macOS: Ollama should start automatically
# Linux:
systemctl start ollama
```

#### Ollama: "Model not found"
```bash
# List available models
ollama list

# Pull the model specified in .env
ollama pull llama2
```

#### LM Studio: "Connection error"
- Ensure LM Studio's local server is started
- Check the port matches your .env configuration
- Verify a model is loaded in LM Studio

#### OpenAI: "Invalid API key"
- Check your API key is correct in .env
- Ensure you have billing enabled on OpenAI account
- Verify the key has the correct permissions

### Document Processing Issues

#### "No text could be extracted"
- Ensure the document is not encrypted or password-protected
- Try converting the document to a different format
- Check if the file is corrupted

#### "File type not supported"
- Currently supported: PDF, DOCX, XLSX, CSV, TXT
- Convert unsupported formats to one of these

### Performance Issues

#### Slow embedding generation
- Use a smaller embedding model in .env:
  ```env
  EMBEDDING_MODEL=all-MiniLM-L12-v2
  ```
- Reduce chunk size:
  ```env
  CHUNK_SIZE=500
  ```

#### Slow LLM responses
- Use a smaller model (e.g., llama2 instead of llama3)
- Reduce max_tokens in .env:
  ```env
  LLM_MAX_TOKENS=500
  ```

#### Out of memory errors
- Close other applications
- Use smaller models
- Reduce batch size for processing

## Advanced Configuration

### Using GPU Acceleration

If you have an NVIDIA GPU:

```bash
# Install CUDA-enabled dependencies
pip uninstall faiss-cpu
pip install faiss-gpu

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Custom Embedding Models

Edit `.env` to use different models:
```env
# Smaller, faster (less accurate)
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Larger, more accurate (slower)
EMBEDDING_MODEL=all-mpnet-base-v2

# Multilingual support
EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2
```

### Production Deployment

For production use:

1. **Use a proper ASGI server**:
   ```bash
   pip install gunicorn
   gunicorn backend.main:app -w 4 -k uvicorn.workers.UvicornWorker
   ```

2. **Add authentication**:
   - Implement JWT tokens
   - Add user management

3. **Enable HTTPS**:
   - Use nginx as reverse proxy
   - Configure SSL certificates

4. **Monitor performance**:
   - Add logging
   - Set up monitoring tools

## Next Steps

1. **Customize the system**:
   - Modify prompts in `rag_pipeline.py`
   - Adjust chunk sizes for your documents
   - Fine-tune retrieval parameters

2. **Add more features**:
   - Implement document summarization
   - Add multi-language support
   - Create custom document parsers

3. **Optimize performance**:
   - Cache frequent queries
   - Implement query preprocessing
   - Use more efficient vector indices

## Getting Help

- **Documentation**: See README.md
- **Issues**: Report on GitHub
- **API Reference**: http://localhost:8000/docs

---

**Congratulations!** 🎉 Your Enterprise Multimodal Assistant is now set up and ready to use!
