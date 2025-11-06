# Quick Reference Guide

## 📝 Common Commands

### Start Application
```bash
# Backend (Terminal 1)
python backend/main.py

# Frontend (Terminal 2)
streamlit run frontend/streamlit_app.py
```

### Using Helper Scripts
```bash
# Windows
.\scripts\start_backend.ps1
.\scripts\start_frontend.ps1

# Linux/Mac
chmod +x scripts/*.sh
./scripts/start_backend.sh
./scripts/start_frontend.sh
```

## 🔗 Important URLs

| Service | URL | Description |
|---------|-----|-------------|
| Frontend UI | http://localhost:8501 | Main user interface |
| Backend API | http://localhost:8000 | REST API |
| API Docs | http://localhost:8000/docs | Interactive API documentation |
| Health Check | http://localhost:8000/health | Server status |

## 🛠️ API Quick Reference

### Upload Document
```bash
curl -X POST -F "file=@document.pdf" http://localhost:8000/upload
```

### Chat Query
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is this about?",
    "top_k": 5
  }'
```

### List Documents
```bash
curl http://localhost:8000/documents
```

### Delete Document
```bash
curl -X DELETE http://localhost:8000/documents/{file_id}
```

### Health Check
```bash
curl http://localhost:8000/health
```

## ⚙️ Configuration Quick Reference

### LLM Providers

#### Ollama (Local)
```env
LLM_PROVIDER=ollama
LLM_MODEL=llama2
LLM_BASE_URL=http://localhost:11434
```

#### LM Studio (Local)
```env
LLM_PROVIDER=lmstudio
LLM_MODEL=mistral-7b
LLM_BASE_URL=http://localhost:1234/v1
```

#### OpenAI (Cloud)
```env
LLM_PROVIDER=openai
LLM_MODEL=gpt-3.5-turbo
OPENAI_API_KEY=sk-...
```

### Popular Embedding Models

| Model | Size | Dimension | Speed | Quality |
|-------|------|-----------|-------|---------|
| all-MiniLM-L6-v2 | 80MB | 384 | Fast | Good |
| all-MiniLM-L12-v2 | 120MB | 384 | Medium | Better |
| all-mpnet-base-v2 | 420MB | 768 | Slow | Best |
| paraphrase-multilingual-MiniLM-L12-v2 | 470MB | 384 | Medium | Multilingual |

## 📚 Supported File Types

| Extension | Type | Library Used |
|-----------|------|--------------|
| .pdf | PDF | PyMuPDF |
| .docx, .doc | Word | python-docx |
| .xlsx, .xls | Excel | pandas + openpyxl |
| .csv | CSV | pandas |
| .txt | Text | built-in |

## 🐛 Troubleshooting Quick Fixes

### Backend won't start
```bash
# Check if port is in use
netstat -ano | findstr :8000  # Windows
lsof -ti:8000  # Linux/Mac

# Kill process
taskkill /PID <PID> /F  # Windows
kill -9 <PID>  # Linux/Mac
```

### Frontend can't connect
```bash
# Verify backend is running
curl http://localhost:8000/health

# Check if frontend is on correct port
netstat -ano | findstr :8501  # Windows
lsof -ti:8501  # Linux/Mac
```

### Ollama not responding
```bash
# Check Ollama status
ollama list

# Restart Ollama
# Windows/Mac: Restart from system tray
# Linux:
systemctl restart ollama
```

### Module not found errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Out of memory
```bash
# Use smaller models
# In .env:
EMBEDDING_MODEL=all-MiniLM-L6-v2
LLM_MODEL=llama2  # or smaller variant

# Reduce chunk size
CHUNK_SIZE=500
```

## 🔍 Useful Ollama Commands

```bash
# List installed models
ollama list

# Pull a model
ollama pull llama2
ollama pull mistral
ollama pull llama3

# Remove a model
ollama rm llama2

# Show model info
ollama show llama2

# Test model
ollama run llama2 "Hello!"
```

## 📊 File Structure at a Glance

```
enterprise-multimodal-assistant/
├── backend/           # FastAPI server
├── frontend/          # Streamlit UI
├── scripts/           # Helper scripts
├── data/             # Auto-created at runtime
├── requirements.txt  # Dependencies
├── .env.example      # Config template
└── README.md         # Documentation
```

## 💡 Common Use Cases

### 1. Process Multiple Documents
```python
# Upload documents via UI, then query across all:
"What are the common themes across all reports?"
"Compare Q1 and Q2 performance"
```

### 2. Extract Specific Information
```python
"List all action items from the meeting notes"
"What are the financial projections for next year?"
```

### 3. Summarization
```python
"Summarize the key findings in the research paper"
"What are the main conclusions?"
```

### 4. Question Answering
```python
"Who is responsible for the marketing budget?"
"When is the project deadline?"
```

## 🎯 Performance Tuning

### For Speed
```env
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHUNK_SIZE=500
LLM_MAX_TOKENS=500
# Use smaller LLM models
```

### For Quality
```env
EMBEDDING_MODEL=all-mpnet-base-v2
CHUNK_SIZE=1000
CHUNK_OVERLAP=300
LLM_TEMPERATURE=0.3  # More focused
# Use larger LLM models
```

### For Large Documents
```env
CHUNK_SIZE=2000
CHUNK_OVERLAP=400
# Consider using GPU acceleration
```

## 🔐 Security Checklist

- [ ] Change default CORS settings in production
- [ ] Implement authentication
- [ ] Use HTTPS
- [ ] Validate all file uploads
- [ ] Set up rate limiting
- [ ] Enable logging
- [ ] Regular security updates

## 📞 Getting Help

- 📖 Full documentation: README.md
- 🔧 Setup guide: SETUP.md
- 📊 Project details: PROJECT_OVERVIEW.md
- 🐛 Issues: GitHub Issues
- 💬 Discussions: GitHub Discussions

## 🎓 Learning Resources

### RAG Concepts
- [Retrieval-Augmented Generation Paper](https://arxiv.org/abs/2005.11401)
- [LangChain RAG Tutorial](https://python.langchain.com/docs/use_cases/question_answering/)

### Technologies
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [FAISS Documentation](https://faiss.ai/)
- [Sentence Transformers](https://www.sbert.net/)

### Local LLMs
- [Ollama Documentation](https://github.com/ollama/ollama)
- [LM Studio Guide](https://lmstudio.ai/docs)

---

**Quick Start**: Run `python backend/main.py` and `streamlit run frontend/streamlit_app.py`
