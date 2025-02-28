import os
import streamlit as st
import tempfile
from typing import List, Dict, Any, Optional
import time

from document_processor import DocumentProcessor
from indexing import DocumentIndexer
from hybrid_search import HybridSearchEngine
from rag_system import RAGSystem

# Configure page
st.set_page_config(page_title="RAG Document Assistant", layout="wide", page_icon="📚")

# Initialize session state variables if they don't exist
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.documents_loaded = False
    st.session_state.chat_history = []
    st.session_state.rag_system = None
    st.session_state.document_info = []


def initialize_system():
    """Initialize the RAG system with models and database."""
    with st.spinner("Initializing models and database..."):
        # For HF Spaces, we can use a lightweight model
        embedding_model = "sentence-transformers/all-MiniLM-L6-v2"  # Smaller model for HF Spaces
        llm_model = "google/flan-t5-small"  # Lightweight LLM for HF Spaces
        
        # Use MongoDB Atlas for persistence (in production) or an in-memory option for demo
        use_mock_db = os.environ.get("USE_MOCK_DB", "true").lower() == "true"
        
        if use_mock_db:
            st.info("Using in-memory database for demo purposes.")
            db_uri = "memory://"
            db_name = "rag_demo"
        else:
            # In production, use MongoDB Atlas connection string from secrets
            db_uri = os.environ.get("MONGODB_URI", "mongodb://localhost:27017/")
            db_name = "rag_db"
        
        # Initialize RAG system
        search_engine = HybridSearchEngine(
            embedding_model_name=embedding_model,
            db_uri=db_uri, 
            db_name=db_name,
            vector_weight=0.7,
            lexical_weight=0.3
        )
        
        rag_system = RAGSystem(
            search_engine=search_engine,
            llm_model_name=llm_model,
            device="cpu"  # Use CPU for HF Spaces
        )
        
        st.session_state.rag_system = rag_system
        st.session_state.initialized = True
        return rag_system


def render_chat_history():
    """Render the chat history in the Streamlit UI."""
    for i, message in enumerate(st.session_state.chat_history):
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])
                
                # If there are sources, display them
                if "sources" in message and message["sources"]:
                    with st.expander("View Sources"):
                        for j, source in enumerate(message["sources"]):
                            st.markdown(f"**Source {j+1}**: {source['text'][:200]}...")
                            st.caption(f"From: {source.get('metadata', {}).get('source', 'Unknown')}")


def process_documents(uploaded_files):
    """Process uploaded documents and index them."""
    if not st.session_state.initialized:
        initialize_system()
        
    rag_system = st.session_state.rag_system
    
    with st.spinner("Processing documents..."):
        doc_info = []
        
        for uploaded_file in uploaded_files:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                file_path = tmp_file.name
            
            # Process and index the document
            try:
                metadata = {
                    "source": uploaded_file.name,
                    "uploaded_at": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
                result = rag_system.add_document(file_path, metadata=metadata)
                
                doc_info.append({
                    "filename": uploaded_file.name,
                    "chunk_count": result["chunk_count"],
                    "status": "Indexed successfully"
                })
                
            except Exception as e:
                doc_info.append({
                    "filename": uploaded_file.name,
                    "chunk_count": 0,
                    "status": f"Error: {str(e)}"
                })
            
            # Clean up temp file
            os.unlink(file_path)
        
        st.session_state.documents_loaded = True
        st.session_state.document_info = doc_info
        
        return doc_info


def main():
    st.title("📚 RAG Document Assistant")
    
    # Sidebar for uploads and settings
    with st.sidebar:
        st.header("Document Upload")
        uploaded_files = st.file_uploader(
            "Upload PDF, TXT, or DOCX files", 
            accept_multiple_files=True,
            type=["pdf", "txt", "docx"]
        )
        
        if uploaded_files and st.button("Process Documents"):
            doc_info = process_documents(uploaded_files)
            
            # Display document processing results
            st.subheader("Processed Documents")
            for doc in doc_info:
                status_color = "green" if "successfully" in doc["status"] else "red"
                st.markdown(f"**{doc['filename']}**: {doc['chunk_count']} chunks - "
                           f"<span style='color:{status_color}'>{doc['status']}</span>", 
                           unsafe_allow_html=True)
        
        # Display indexed documents
        if st.session_state.document_info:
            st.subheader("Indexed Documents")
            for doc in st.session_state.document_info:
                st.text(f"• {doc['filename']}")
        
        st.divider()
        
        # Settings
        st.subheader("Settings")
        vector_weight = st.slider("Vector Search Weight", 0.0, 1.0, 0.7, 0.1)
        lexical_weight = st.slider("Lexical Search Weight", 0.0, 1.0, 0.3, 0.1)
        
        if st.button("Apply Settings"):
            if st.session_state.initialized:
                st.session_state.rag_system.search_engine.vector_weight = vector_weight
                st.session_state.rag_system.search_engine.lexical_weight = lexical_weight
                st.success("Settings applied!")
    
    # Initialize system if not done yet
    if not st.session_state.initialized:
        initialize_system()
    
    # Main chat interface
    render_chat_history()
    
    # Chat input
    if query := st.chat_input("Ask a question about your documents..."):
        if not st.session_state.documents_loaded:
            st.error("Please upload and process documents first!")
            return
        
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": query})
        
        # Display user message
        with st.chat_message("user"):
            st.write(query)
        
        # Get response from RAG system
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.rag_system.query(query)
                response = result["response"]
                sources = result["sources"]
                
                # Display response
                st.write(response)
                
                # Display sources if available
                if sources:
                    with st.expander("View Sources"):
                        for i, source in enumerate(sources):
                            st.markdown(f"**Source {i+1}**: {source['text'][:200]}...")
                            st.caption(f"From: {source.get('metadata', {}).get('source', 'Unknown')}")
        
        # Add assistant response to chat history
        st.session_state.chat_history.append({
            "role": "assistant", 
            "content": response,
            "sources": sources
        })


if __name__ == "__main__":
    main()