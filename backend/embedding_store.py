"""
Embedding Store Module
Manages vector embeddings using FAISS for efficient similarity search.
Handles document indexing, retrieval, and metadata management.
"""

import numpy as np
import faiss
import pickle
from typing import List, Dict, Optional
from pathlib import Path
from sentence_transformers import SentenceTransformer
from datetime import datetime


class EmbeddingStore:
    """
    Vector database implementation using FAISS for local embedding storage.
    """
    
    def __init__(self, db_path: str = "./data/vector_db", 
                 model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding store with FAISS index and sentence transformer.
        
        Args:
            db_path: Path to store FAISS index and metadata
            model_name: Sentence transformer model name for embeddings
        """
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        self.model_name = model_name
        self.embedding_model = SentenceTransformer(model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize or load FAISS index
        self.index_path = self.db_path / "faiss.index"
        self.metadata_path = self.db_path / "metadata.pkl"
        
        self.index = self._load_or_create_index()
        self.metadata = self._load_or_create_metadata()
    
    def _load_or_create_index(self) -> faiss.Index:
        """
        Load existing FAISS index or create a new one.
        
        Returns:
            faiss.Index: FAISS index for similarity search
        """
        if self.index_path.exists():
            print(f"Loading existing FAISS index from {self.index_path}")
            return faiss.read_index(str(self.index_path))
        else:
            print(f"Creating new FAISS index with dimension {self.embedding_dim}")
            # Using IndexFlatL2 for exact search (can be optimized with IndexIVFFlat for larger datasets)
            index = faiss.IndexFlatL2(self.embedding_dim)
            return index
    
    def _load_or_create_metadata(self) -> List[Dict]:
        """
        Load existing metadata or create empty list.
        
        Returns:
            list: Metadata for each indexed document chunk
        """
        if self.metadata_path.exists():
            print(f"Loading metadata from {self.metadata_path}")
            with open(self.metadata_path, "rb") as f:
                return pickle.load(f)
        else:
            print("Creating new metadata store")
            return []
    
    def _save_index(self):
        """Save FAISS index to disk"""
        faiss.write_index(self.index, str(self.index_path))
        print(f"Saved FAISS index to {self.index_path}")
    
    def _save_metadata(self):
        """Save metadata to disk"""
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"Saved metadata to {self.metadata_path}")
    
    def add_documents(self, texts: List[str], metadata: Dict):
        """
        Add documents to the vector database.
        
        Args:
            texts: List of text chunks to embed and store
            metadata: Metadata dictionary (file_id, filename, etc.)
        """
        if not texts:
            return
        
        print(f"Generating embeddings for {len(texts)} text chunks...")
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(
            texts, 
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Normalize embeddings for cosine similarity (optional but recommended)
        faiss.normalize_L2(embeddings)
        
        # Add to FAISS index
        self.index.add(embeddings)
        
        # Store metadata for each chunk
        timestamp = datetime.now().isoformat()
        for idx, text in enumerate(texts):
            chunk_metadata = {
                **metadata,
                "chunk_id": len(self.metadata) + idx,
                "text": text,
                "timestamp": timestamp
            }
            self.metadata.append(chunk_metadata)
        
        # Save to disk
        self._save_index()
        self._save_metadata()
        
        print(f"Successfully added {len(texts)} chunks to vector database")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for similar documents using semantic similarity.
        
        Args:
            query: Query string to search for
            top_k: Number of top results to return
            
        Returns:
            list: Retrieved documents with metadata and similarity scores
        """
        if self.index.ntotal == 0:
            print("Warning: Vector database is empty")
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True
        )
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search in FAISS index
        top_k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Prepare results with metadata
        results = []
        for idx, (distance, doc_idx) in enumerate(zip(distances[0], indices[0])):
            if doc_idx < len(self.metadata):
                result = {
                    "text": self.metadata[doc_idx]["text"],
                    "metadata": {
                        "filename": self.metadata[doc_idx].get("filename", "Unknown"),
                        "file_id": self.metadata[doc_idx].get("file_id", "Unknown"),
                        "chunk_id": self.metadata[doc_idx].get("chunk_id", doc_idx),
                    },
                    "score": float(distance),
                    "rank": idx + 1
                }
                results.append(result)
        
        return results
    
    def delete_by_file_id(self, file_id: str) -> int:
        """
        Delete all embeddings associated with a file ID.
        Note: FAISS doesn't support efficient deletion, so we rebuild the index.
        
        Args:
            file_id: File ID to delete
            
        Returns:
            int: Number of chunks deleted
        """
        # Find indices to keep
        indices_to_keep = []
        new_metadata = []
        
        for idx, meta in enumerate(self.metadata):
            if meta.get("file_id") != file_id:
                indices_to_keep.append(idx)
                new_metadata.append(meta)
        
        deleted_count = len(self.metadata) - len(new_metadata)
        
        if deleted_count == 0:
            return 0
        
        # Rebuild index with remaining embeddings
        print(f"Rebuilding index after deleting {deleted_count} chunks...")
        
        # Create new index
        new_index = faiss.IndexFlatL2(self.embedding_dim)
        
        # Extract embeddings for kept documents
        if indices_to_keep:
            # Reconstruct vectors from old index
            kept_embeddings = np.zeros((len(indices_to_keep), self.embedding_dim), dtype=np.float32)
            for new_idx, old_idx in enumerate(indices_to_keep):
                kept_embeddings[new_idx] = self.index.reconstruct(old_idx)
            
            new_index.add(kept_embeddings)
        
        # Update index and metadata
        self.index = new_index
        self.metadata = new_metadata
        
        # Save to disk
        self._save_index()
        self._save_metadata()
        
        print(f"Deleted {deleted_count} chunks from vector database")
        return deleted_count
    
    def get_document_count(self) -> int:
        """
        Get the number of unique documents indexed.
        
        Returns:
            int: Number of unique documents
        """
        file_ids = set(meta.get("file_id") for meta in self.metadata)
        return len(file_ids)
    
    def get_vector_count(self) -> int:
        """
        Get the total number of vectors in the index.
        
        Returns:
            int: Number of vectors
        """
        return self.index.ntotal
    
    def list_documents(self) -> List[Dict]:
        """
        List all indexed documents with metadata.
        
        Returns:
            list: List of documents with metadata
        """
        # Group by file_id
        documents = {}
        
        for meta in self.metadata:
            file_id = meta.get("file_id", "Unknown")
            if file_id not in documents:
                documents[file_id] = {
                    "file_id": file_id,
                    "filename": meta.get("filename", "Unknown"),
                    "chunk_count": 0,
                    "timestamp": meta.get("timestamp", "Unknown")
                }
            documents[file_id]["chunk_count"] += 1
        
        return list(documents.values())
    
    def clear_all(self):
        """
        Clear all embeddings and metadata from the database.
        """
        print("Clearing all data from vector database...")
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.metadata = []
        self._save_index()
        self._save_metadata()
        print("Vector database cleared")
