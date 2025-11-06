"""
Streamlit Frontend for Enterprise Multimodal Assistant
Provides a chat interface with document upload capabilities.
"""

import streamlit as st
import requests
from typing import List, Dict
import time
from pathlib import Path

# Configuration
API_BASE_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="Enterprise Multimodal Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f5f5f5;
        border-left: 4px solid #4caf50;
    }
    .source-box {
        background-color: #fff3e0;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin-top: 0.5rem;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


def check_backend_health():
    """Check if backend server is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except requests.exceptions.RequestException:
        return False, None


def upload_document(file):
    """Upload a document to the backend"""
    try:
        files = {"file": (file.name, file.getvalue(), file.type)}
        response = requests.post(f"{API_BASE_URL}/upload", files=files, timeout=60)
        response.raise_for_status()
        return True, response.json()
    except requests.exceptions.RequestException as e:
        return False, str(e)


def send_chat_message(query: str, conversation_id: str = None, top_k: int = 5):
    """Send a chat message to the backend"""
    try:
        payload = {
            "query": query,
            "conversation_id": conversation_id,
            "top_k": top_k
        }
        response = requests.post(f"{API_BASE_URL}/chat", json=payload, timeout=60)
        response.raise_for_status()
        return True, response.json()
    except requests.exceptions.RequestException as e:
        return False, str(e)


def get_documents():
    """Get list of uploaded documents"""
    try:
        response = requests.get(f"{API_BASE_URL}/documents", timeout=10)
        response.raise_for_status()
        return True, response.json()
    except requests.exceptions.RequestException as e:
        return False, str(e)


def delete_document(file_id: str):
    """Delete a document from the backend"""
    try:
        response = requests.delete(f"{API_BASE_URL}/documents/{file_id}", timeout=10)
        response.raise_for_status()
        return True, response.json()
    except requests.exceptions.RequestException as e:
        return False, str(e)


def init_session_state():
    """Initialize session state variables"""
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "backend_status" not in st.session_state:
        st.session_state.backend_status = None


def render_chat_message(role: str, content: str, sources: List[Dict] = None):
    """Render a chat message with sources"""
    css_class = "user-message" if role == "user" else "assistant-message"
    icon = "👤" if role == "user" else "🤖"
    
    st.markdown(f"""
    <div class="chat-message {css_class}">
        <strong>{icon} {role.capitalize()}:</strong><br>
        {content}
    </div>
    """, unsafe_allow_html=True)
    
    # Show sources for assistant messages
    if sources and role == "assistant":
        with st.expander("📚 View Sources", expanded=False):
            for idx, source in enumerate(sources, 1):
                st.markdown(f"""
                <div class="source-box">
                    <strong>Source {idx}: {source['filename']}</strong><br>
                    <em>Relevance Score: {source['relevance_score']:.4f}</em><br>
                    {source['text_preview']}
                </div>
                """, unsafe_allow_html=True)


def main():
    """Main application"""
    init_session_state()
    
    # Header
    st.markdown('<div class="main-header">🤖 Enterprise Multimodal Assistant</div>', unsafe_allow_html=True)
    
    # Check backend status
    is_healthy, health_data = check_backend_health()
    
    if not is_healthy:
        st.error("⚠️ Backend server is not running. Please start the backend with: `uvicorn backend.main:app --reload`")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Display backend status
        st.success("✅ Backend Connected")
        if health_data:
            st.info(f"""
            **Embedding Model:** {health_data.get('embedding_model', 'Unknown')}  
            **Documents Indexed:** {health_data.get('documents_indexed', 0)}  
            **Vector Count:** {health_data.get('vector_db_size', 0)}
            """)
        
        st.divider()
        
        # Document upload
        st.header("📄 Upload Documents")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['pdf', 'docx', 'doc', 'xlsx', 'xls', 'csv', 'txt'],
            help="Upload PDF, Word, Excel, CSV, or text files"
        )
        
        if uploaded_file:
            if st.button("🚀 Upload & Process", type="primary"):
                with st.spinner("Processing document..."):
                    success, result = upload_document(uploaded_file)
                    if success:
                        st.success(f"✅ {result['message']}")
                        st.info(f"Created {result['chunks_created']} text chunks")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(f"❌ Upload failed: {result}")
        
        st.divider()
        
        # Document management
        st.header("📚 Manage Documents")
        success, doc_data = get_documents()
        
        if success and doc_data.get('documents'):
            for doc in doc_data['documents']:
                with st.expander(f"📄 {doc['filename']}", expanded=False):
                    st.text(f"File ID: {doc['file_id']}")
                    st.text(f"Chunks: {doc['chunk_count']}")
                    st.text(f"Uploaded: {doc['timestamp'][:19]}")
                    
                    if st.button(f"🗑️ Delete", key=f"delete_{doc['file_id']}"):
                        success, result = delete_document(doc['file_id'])
                        if success:
                            st.success("Document deleted!")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(f"Delete failed: {result}")
        else:
            st.info("No documents uploaded yet")
        
        st.divider()
        
        # Chat settings
        st.header("💬 Chat Settings")
        top_k = st.slider(
            "Number of sources to retrieve",
            min_value=1,
            max_value=10,
            value=5,
            help="How many relevant document chunks to retrieve"
        )
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.session_state.conversation_id = None
            st.rerun()
    
    # Main chat interface
    st.header("💬 Chat Interface")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            render_chat_message(
                role=message["role"],
                content=message["content"],
                sources=message.get("sources")
            )
    
    # Chat input
    st.divider()
    
    # Create columns for input and button
    col1, col2 = st.columns([5, 1])
    
    with col1:
        user_input = st.text_input(
            "Ask a question about your documents",
            key="user_input",
            placeholder="e.g., What are the key findings in the Q4 report?",
            label_visibility="collapsed"
        )
    
    with col2:
        send_button = st.button("Send", type="primary", use_container_width=True)
    
    # Process user input
    if send_button and user_input:
        # Add user message to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input,
            "sources": None
        })
        
        # Send to backend
        with st.spinner("🤔 Thinking..."):
            success, result = send_chat_message(
                query=user_input,
                conversation_id=st.session_state.conversation_id,
                top_k=top_k
            )
            
            if success:
                # Update conversation ID
                st.session_state.conversation_id = result['conversation_id']
                
                # Add assistant response to history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": result['response'],
                    "sources": result['sources']
                })
                
                st.rerun()
            else:
                st.error(f"❌ Error: {result}")
    
    # Instructions for first-time users
    if not st.session_state.chat_history:
        st.info("""
        👋 **Welcome to the Enterprise Multimodal Assistant!**
        
        **Getting Started:**
        1. Upload your documents using the sidebar (PDF, Word, Excel, CSV, or text files)
        2. Wait for the documents to be processed and indexed
        3. Ask questions about your documents in the chat box below
        
        **Features:**
        - 🔒 **Fully Local**: All data stays on your machine
        - 🧠 **Context-Aware**: Uses RAG to provide accurate answers
        - 📚 **Multi-Document**: Query across all your uploaded documents
        - 🔍 **Source Citations**: See exactly where answers come from
        """)


if __name__ == "__main__":
    main()
