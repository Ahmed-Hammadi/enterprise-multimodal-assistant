# Enterprise Multimodal Assistant 🤖

A fully local, privacy-first enterprise assistant built with Retrieval-Augmented Generation (RAG). Process and query internal company documents (PDFs, Excel, Word, reports) while ensuring complete data privacy by running entirely offline.

## 🎯 Features

- **🔒 Fully Local & Private**: No cloud storage, all data stays on your machine
- **📚 Multi-Format Support**: Process PDFs, Word documents, Excel files, CSV, and text files
- **🧠 Context-Aware RAG**: Retrieval-Augmented Generation for accurate, grounded answers
- **💬 Multi-Turn Conversations**: Maintains conversation history for coherent dialogue
- **🔍 Source Citations**: See exactly which documents informed each answer
- **⚡ Fast Retrieval**: FAISS-powered vector search for efficient document retrieval
- **🎨 User-Friendly Interface**: Clean Streamlit UI for document upload and chat

## 🏗️ Architecture

```
┌─────────────────┐
│   Streamlit UI  │  ← User Interface
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   FastAPI       │  ← REST API
│   Backend       │
└────────┬────────┘
         │
         ├─────────────────────┐
         │                     │
         ▼                     ▼
┌─────────────────┐   ┌─────────────────┐
│  Document       │   │  RAG Pipeline   │
│  Processor      │   │                 │
└─────────────────┘   └────────┬────────┘
                               │
                     ┌─────────┴─────────┐
                     │                   │
                     ▼                   ▼
            ┌─────────────────┐  ┌─────────────────┐
            │  FAISS Vector   │  │  Local LLM      │
            │  Database       │  │  (Ollama/LM     │
            │                 │  │  Studio/OpenAI) │
            └─────────────────┘  └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- A local LLM (optional but recommended):
  - [Ollama](https://ollama.ai/) (recommended)
  - [LM Studio](https://lmstudio.ai/)
  - OpenAI API key

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Ahmed-Hammadi/enterprise-multimodal-assistant.git
   cd enterprise-multimodal-assistant
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   .\venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment** (optional)
   ```bash
   cp .env.example .env
   # Edit .env to configure your LLM provider
   ```

### Running the Application

1. **Start the Backend Server**
   ```bash
   cd backend
   python main.py
   ```
   Or using uvicorn directly:
   ```bash
   uvicorn backend.main:app --reload
   ```
   
   The API will be available at `http://localhost:8000`
   
   API Documentation: `http://localhost:8000/docs`

2. **Start the Frontend** (in a new terminal)
   ```bash
   streamlit run frontend/streamlit_app.py
   ```
   
   The UI will open automatically in your browser at `http://localhost:8501`

### Setting Up a Local LLM

#### Option 1: Ollama (Recommended)

1. Install Ollama from [ollama.ai](https://ollama.ai/)
2. Pull a model:
   ```bash
   ollama pull llama2
   ```
3. Configure in `.env`:
   ```env
   LLM_PROVIDER=ollama
   LLM_MODEL=llama2
   LLM_BASE_URL=http://localhost:11434
   ```

#### Option 2: LM Studio

1. Download LM Studio from [lmstudio.ai](https://lmstudio.ai/)
2. Download a model (e.g., Mistral-7B)
3. Start the local server
4. Configure in `.env`:
   ```env
   LLM_PROVIDER=lmstudio
   LLM_MODEL=mistral-7b
   LLM_BASE_URL=http://localhost:1234/v1
   ```

#### Option 3: OpenAI API

Configure in `.env`:
```env
LLM_PROVIDER=openai
LLM_MODEL=gpt-3.5-turbo
OPENAI_API_KEY=your_api_key_here
```

## 📖 Usage

### Upload Documents

1. Use the sidebar to upload documents (PDF, Word, Excel, CSV, or text files)
2. Click "Upload & Process" to index the document
3. Wait for processing to complete (embeddings are generated)

### Chat with Your Documents

1. Type your question in the chat input box
2. The system will:
   - Retrieve relevant document chunks
   - Assemble context with conversation history
   - Query the LLM for a grounded response
3. View sources and relevance scores by expanding the "View Sources" section

### Manage Documents

- View all uploaded documents in the sidebar
- Delete documents you no longer need
- Monitor the number of indexed chunks

## 🧩 Project Structure

```
enterprise-multimodal-assistant/
├── backend/
│   ├── main.py                 # FastAPI application with endpoints
│   ├── rag_pipeline.py         # RAG orchestration and LLM integration
│   ├── embedding_store.py      # FAISS vector database management
│   └── utils/
│       ├── __init__.py
│       └── document_processor.py  # Document extraction utilities
├── frontend/
│   └── streamlit_app.py        # Streamlit UI application
├── data/                       # Created at runtime
│   ├── uploads/                # Uploaded documents
│   └── vector_db/              # FAISS index and metadata
├── requirements.txt            # Python dependencies
├── .env.example               # Environment configuration template
├── .gitignore
└── README.md
```

## 🔧 API Endpoints

### Health Check
```http
GET /health
```
Returns server status and statistics.

### Upload Document
```http
POST /upload
Content-Type: multipart/form-data

file: <document_file>
```
Uploads and processes a document.

### Chat Query
```http
POST /chat
Content-Type: application/json

{
  "query": "What are the key findings?",
  "conversation_id": "optional-conversation-id",
  "top_k": 5
}
```
Processes a chat query using RAG.

### List Documents
```http
GET /documents
```
Returns list of all uploaded documents.

### Delete Document
```http
DELETE /documents/{file_id}
```
Deletes a document and its embeddings.

## 🔐 Security & Privacy

- **No Cloud Storage**: All documents and embeddings stored locally
- **Local Processing**: Document processing happens on your machine
- **Local LLM Support**: Use Ollama or LM Studio for complete privacy
- **No Data Leakage**: Nothing sent to external services (except when using OpenAI)

## 🎛️ Configuration

Key configuration options in `.env`:

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | LLM provider (ollama, lmstudio, openai) | ollama |
| `LLM_MODEL` | Model name | llama2 |
| `LLM_BASE_URL` | LLM endpoint URL | http://localhost:11434 |
| `EMBEDDING_MODEL` | Sentence transformer model | all-MiniLM-L6-v2 |
| `CHUNK_SIZE` | Text chunk size in characters | 1000 |
| `CHUNK_OVERLAP` | Overlap between chunks | 200 |

## 🧪 Testing

Test the backend API:
```bash
# Health check
curl http://localhost:8000/health

# Upload document
curl -X POST -F "file=@document.pdf" http://localhost:8000/upload

# Chat query
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is this document about?"}'
```

## 🛠️ Troubleshooting

### Backend won't start
- Ensure port 8000 is not in use
- Check Python version (3.8+ required)
- Verify all dependencies are installed

### Documents won't upload
- Check file format is supported
- Ensure file is not corrupted
- Check backend logs for specific errors

### LLM not responding
- Verify LLM service is running (Ollama/LM Studio)
- Check LLM_BASE_URL in .env
- Test LLM endpoint independently

### Slow performance
- Use GPU-enabled sentence-transformers if available
- Reduce CHUNK_SIZE for faster processing
- Use smaller embedding models
- Consider using IndexIVFFlat instead of IndexFlatL2 for large datasets

## 📚 Technologies Used

- **Backend**: FastAPI, Python
- **Frontend**: Streamlit
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Database**: FAISS
- **Document Processing**: PyMuPDF, python-docx, pandas
- **LLM Integration**: Ollama, LM Studio, OpenAI

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [FAISS](https://github.com/facebookresearch/faiss) for efficient similarity search
- [Sentence Transformers](https://www.sbert.net/) for embeddings
- [Ollama](https://ollama.ai/) for easy local LLM deployment
- [FastAPI](https://fastapi.tiangolo.com/) for the robust API framework
- [Streamlit](https://streamlit.io/) for the intuitive UI

## 📧 Contact

Ahmed Hammadi - [@Ahmed-Hammadi](https://github.com/Ahmed-Hammadi)

Project Link: [https://github.com/Ahmed-Hammadi/enterprise-multimodal-assistant](https://github.com/Ahmed-Hammadi/enterprise-multimodal-assistant)

---

⭐ If you find this project useful, please consider giving it a star!