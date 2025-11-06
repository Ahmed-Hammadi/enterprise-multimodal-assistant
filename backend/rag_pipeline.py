"""
RAG Pipeline Module
Handles retrieval-augmented generation workflow:
1. Retrieve relevant document chunks
2. Assemble context with conversation history
3. Query LLM with enriched context
"""

from typing import List, Dict, Optional
import os
from datetime import datetime


class RAGPipeline:
    """
    Orchestrates the RAG workflow for context-aware query answering.
    """
    
    def __init__(self, embedding_store, llm_config: Optional[Dict] = None):
        """
        Initialize RAG pipeline with embedding store and LLM configuration.
        
        Args:
            embedding_store: EmbeddingStore instance for vector retrieval
            llm_config: Configuration for LLM (endpoint, model name, etc.)
        """
        self.embedding_store = embedding_store
        self.llm_config = llm_config or self._default_llm_config()
        self.conversation_history: Dict[str, List[Dict]] = {}
        
        # Initialize LLM client based on configuration
        self.llm_client = self._initialize_llm()
    
    def _default_llm_config(self) -> Dict:
        """
        Default LLM configuration supporting multiple backends.
        
        Returns:
            dict: Default LLM configuration
        """
        return {
            "provider": os.getenv("LLM_PROVIDER", "ollama"),  # ollama, lmstudio, openai
            "model": os.getenv("LLM_MODEL", "llama2"),
            "base_url": os.getenv("LLM_BASE_URL", "http://localhost:11434"),
            "temperature": float(os.getenv("LLM_TEMPERATURE", "0.7")),
            "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "1000")),
        }
    
    def _initialize_llm(self):
        """
        Initialize LLM client based on provider configuration.
        
        Returns:
            LLM client instance
        """
        provider = self.llm_config["provider"].lower()
        
        if provider == "ollama":
            return self._initialize_ollama()
        elif provider == "lmstudio":
            return self._initialize_lmstudio()
        elif provider == "openai":
            return self._initialize_openai()
        else:
            # Fallback to mock LLM for testing
            return self._initialize_mock_llm()
    
    def _initialize_ollama(self):
        """Initialize Ollama client"""
        try:
            import requests
            base_url = self.llm_config["base_url"]
            
            class OllamaClient:
                def __init__(self, base_url, model):
                    self.base_url = base_url.rstrip('/')
                    self.model = model
                
                def generate(self, prompt, temperature=0.7, max_tokens=1000):
                    response = requests.post(
                        f"{self.base_url}/api/generate",
                        json={
                            "model": self.model,
                            "prompt": prompt,
                            "stream": False,
                            "options": {
                                "temperature": temperature,
                                "num_predict": max_tokens
                            }
                        },
                        timeout=60
                    )
                    response.raise_for_status()
                    return response.json()["response"]
            
            return OllamaClient(base_url, self.llm_config["model"])
        except ImportError:
            print("Warning: requests library not available for Ollama")
            return self._initialize_mock_llm()
    
    def _initialize_lmstudio(self):
        """Initialize LM Studio client (OpenAI-compatible API)"""
        try:
            from openai import OpenAI
            
            client = OpenAI(
                base_url=self.llm_config["base_url"],
                api_key="lm-studio"  # LM Studio doesn't require real API key
            )
            
            class LMStudioClient:
                def __init__(self, client, model):
                    self.client = client
                    self.model = model
                
                def generate(self, prompt, temperature=0.7, max_tokens=1000):
                    response = self.client.completions.create(
                        model=self.model,
                        prompt=prompt,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    return response.choices[0].text
            
            return LMStudioClient(client, self.llm_config["model"])
        except ImportError:
            print("Warning: openai library not available for LM Studio")
            return self._initialize_mock_llm()
    
    def _initialize_openai(self):
        """Initialize OpenAI client"""
        try:
            from openai import OpenAI
            
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("Warning: OPENAI_API_KEY not set, using mock LLM")
                return self._initialize_mock_llm()
            
            client = OpenAI(api_key=api_key)
            
            class OpenAIClient:
                def __init__(self, client, model):
                    self.client = client
                    self.model = model
                
                def generate(self, prompt, temperature=0.7, max_tokens=1000):
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    return response.choices[0].message.content
            
            return OpenAIClient(client, self.llm_config["model"])
        except ImportError:
            print("Warning: openai library not available")
            return self._initialize_mock_llm()
    
    def _initialize_mock_llm(self):
        """Initialize mock LLM for testing without actual LLM"""
        class MockLLM:
            def generate(self, prompt, temperature=0.7, max_tokens=1000):
                return (
                    "This is a mock response. To use a real LLM, please configure one of the following:\n"
                    "- Ollama: Set LLM_PROVIDER=ollama and install Ollama locally\n"
                    "- LM Studio: Set LLM_PROVIDER=lmstudio and run LM Studio server\n"
                    "- OpenAI: Set LLM_PROVIDER=openai and OPENAI_API_KEY environment variable\n\n"
                    f"Your query context:\n{prompt[:500]}..."
                )
        
        print("Warning: Using mock LLM. Configure a real LLM provider for production use.")
        return MockLLM()
    
    def _build_context(self, query: str, retrieved_docs: List[Dict], 
                       conversation_id: Optional[str] = None) -> str:
        """
        Build context for LLM by combining retrieved documents and conversation history.
        
        Args:
            query: User query
            retrieved_docs: Retrieved document chunks with metadata
            conversation_id: Optional conversation ID for multi-turn context
            
        Returns:
            str: Formatted context for LLM
        """
        context_parts = []
        
        # Add conversation history if available
        if conversation_id and conversation_id in self.conversation_history:
            history = self.conversation_history[conversation_id][-3:]  # Last 3 turns
            if history:
                context_parts.append("=== Previous Conversation ===")
                for turn in history:
                    context_parts.append(f"User: {turn['query']}")
                    context_parts.append(f"Assistant: {turn['response']}\n")
        
        # Add retrieved documents
        if retrieved_docs:
            context_parts.append("=== Relevant Information from Documents ===")
            for idx, doc in enumerate(retrieved_docs, 1):
                context_parts.append(f"\n[Source {idx}: {doc['metadata'].get('filename', 'Unknown')}]")
                context_parts.append(doc['text'])
        
        # Add current query
        context_parts.append("\n=== Current Question ===")
        context_parts.append(query)
        
        return "\n".join(context_parts)
    
    def _create_prompt(self, context: str) -> str:
        """
        Create prompt for LLM with instructions and context.
        
        Args:
            context: Assembled context string
            
        Returns:
            str: Complete prompt for LLM
        """
        prompt = f"""You are an intelligent enterprise assistant with access to company documents. 
Your task is to answer questions based on the provided context from internal documents.

Instructions:
- Answer the question using ONLY the information provided in the context
- If the context doesn't contain enough information, acknowledge this limitation
- Cite specific sources when making claims
- Be concise but comprehensive
- If previous conversation is provided, maintain context continuity

{context}

=== Your Response ===
Please provide a well-structured answer based on the information above:
"""
        return prompt
    
    def query(self, query: str, conversation_id: Optional[str] = None, 
              top_k: int = 5) -> Dict:
        """
        Process a query through the RAG pipeline.
        
        Args:
            query: User query string
            conversation_id: Optional conversation ID for multi-turn dialogue
            top_k: Number of document chunks to retrieve
            
        Returns:
            dict: Response containing answer, sources, and retrieved chunks
        """
        # Step 1: Retrieve relevant documents
        retrieved_docs = self.embedding_store.search(query, top_k=top_k)
        
        # Step 2: Build context
        context = self._build_context(query, retrieved_docs, conversation_id)
        
        # Step 3: Create prompt
        prompt = self._create_prompt(context)
        
        # Step 4: Generate response
        response = self.llm_client.generate(
            prompt=prompt,
            temperature=self.llm_config["temperature"],
            max_tokens=self.llm_config["max_tokens"]
        )
        
        # Step 5: Store in conversation history
        if conversation_id:
            if conversation_id not in self.conversation_history:
                self.conversation_history[conversation_id] = []
            
            self.conversation_history[conversation_id].append({
                "query": query,
                "response": response,
                "timestamp": datetime.now().isoformat()
            })
        
        # Step 6: Prepare sources information
        sources = [
            {
                "filename": doc["metadata"].get("filename", "Unknown"),
                "relevance_score": float(doc.get("score", 0.0)),
                "text_preview": doc["text"][:200] + "..." if len(doc["text"]) > 200 else doc["text"]
            }
            for doc in retrieved_docs
        ]
        
        return {
            "response": response,
            "sources": sources,
            "retrieved_chunks": [doc["text"] for doc in retrieved_docs]
        }
    
    def clear_conversation(self, conversation_id: str):
        """
        Clear conversation history for a given conversation ID.
        
        Args:
            conversation_id: Conversation ID to clear
        """
        if conversation_id in self.conversation_history:
            del self.conversation_history[conversation_id]
