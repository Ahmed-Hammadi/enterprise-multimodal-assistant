"""
FastAPI Backend for Enterprise Multimodal Assistant
Handles document uploads, vector storage, and RAG-based chat queries.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import uuid
from pathlib import Path

from rag_pipeline import RAGPipeline
from embedding_store import EmbeddingStore
from utils.document_processor import DocumentProcessor

# Initialize FastAPI app
app = FastAPI(
    title="Enterprise Multimodal Assistant",
    description="Local RAG-based assistant for enterprise documents",
    version="1.0.0"
)

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
UPLOAD_DIR = Path("./data/uploads")
VECTOR_DB_PATH = "./data/vector_db"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

embedding_store = EmbeddingStore(db_path=VECTOR_DB_PATH)
rag_pipeline = RAGPipeline(embedding_store=embedding_store)
document_processor = DocumentProcessor()


# Pydantic models for request/response
class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    query: str
    conversation_id: Optional[str] = None
    top_k: int = 5


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    response: str
    conversation_id: str
    sources: List[dict]
    retrieved_chunks: List[str]


class UploadResponse(BaseModel):
    """Response model for upload endpoint"""
    message: str
    file_id: str
    filename: str
    chunks_created: int


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    embedding_model: str
    documents_indexed: int
    vector_db_size: int


# API Endpoints

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint to verify server status and system statistics.
    
    Returns:
        HealthResponse: Server status and system metrics
    """
    try:
        doc_count = embedding_store.get_document_count()
        vector_count = embedding_store.get_vector_count()
        
        return HealthResponse(
            status="healthy",
            embedding_model=embedding_store.model_name,
            documents_indexed=doc_count,
            vector_db_size=vector_count
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a document (PDF, Excel, Word).
    Extracts text, generates embeddings, and stores in vector database.
    
    Args:
        file: Uploaded file (PDF, DOCX, XLSX, CSV)
        
    Returns:
        UploadResponse: Upload status and document metadata
    """
    try:
        # Validate file type
        allowed_extensions = {'.pdf', '.docx', '.doc', '.xlsx', '.xls', '.csv', '.txt'}
        file_ext = Path(file.filename).suffix.lower()
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Generate unique file ID
        file_id = str(uuid.uuid4())
        file_path = UPLOAD_DIR / f"{file_id}_{file.filename}"
        
        # Save uploaded file
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Process document and extract text
        print(f"Processing document: {file.filename}")
        text_chunks = document_processor.process_file(str(file_path), file.filename)
        
        if not text_chunks:
            raise HTTPException(
                status_code=400,
                detail="No text could be extracted from the document"
            )
        
        # Generate embeddings and store in vector database
        print(f"Generating embeddings for {len(text_chunks)} chunks")
        embedding_store.add_documents(
            texts=text_chunks,
            metadata={
                "file_id": file_id,
                "filename": file.filename,
                "file_path": str(file_path)
            }
        )
        
        return UploadResponse(
            message="Document uploaded and processed successfully",
            file_id=file_id,
            filename=file.filename,
            chunks_created=len(text_chunks)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Handle chat queries using RAG pipeline.
    Retrieves relevant context and generates response using LLM.
    
    Args:
        request: ChatRequest containing user query and optional conversation ID
        
    Returns:
        ChatResponse: LLM response with sources and retrieved context
    """
    try:
        if not request.query or not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Generate or use existing conversation ID
        conversation_id = request.conversation_id or str(uuid.uuid4())
        
        # Run RAG pipeline
        print(f"Processing query: {request.query[:100]}...")
        result = rag_pipeline.query(
            query=request.query,
            conversation_id=conversation_id,
            top_k=request.top_k
        )
        
        return ChatResponse(
            response=result["response"],
            conversation_id=conversation_id,
            sources=result["sources"],
            retrieved_chunks=result["retrieved_chunks"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat query failed: {str(e)}")


@app.delete("/documents/{file_id}")
async def delete_document(file_id: str):
    """
    Delete a document and its embeddings from the system.
    
    Args:
        file_id: Unique identifier of the document to delete
        
    Returns:
        dict: Deletion status message
    """
    try:
        # Remove from vector database
        deleted_count = embedding_store.delete_by_file_id(file_id)
        
        # Remove file from disk
        for file_path in UPLOAD_DIR.glob(f"{file_id}_*"):
            file_path.unlink()
        
        return {
            "message": "Document deleted successfully",
            "file_id": file_id,
            "vectors_deleted": deleted_count
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")


@app.get("/documents")
async def list_documents():
    """
    List all uploaded documents with metadata.
    
    Returns:
        dict: List of documents with metadata
    """
    try:
        documents = embedding_store.list_documents()
        return {"documents": documents, "count": len(documents)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
