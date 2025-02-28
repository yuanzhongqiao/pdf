"""
Streamlit UI for the RAG system.
"""

import os
import streamlit as st
import tempfile
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
from config import get_logging_config
import logging.config
logging.config.dictConfig(get_logging_config())
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="RAG Document QA System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "document_count" not in st.session_state:
    st.session_state.document_count = 0
if "initialized" not in st.session_state:
    st.session_state.initialized = False

# Initialize RAG engine
@st.cache_resource
def initialize_rag_engine():
    """Initialize RAG engine."""
    from embedding.model import create_embedding_model
    from storage.vector_db import create_vector_database
    from rag.engine import create_rag_engine
    
    # Create components
    embedding_model = create_embedding_model()
    vector_db = create_vector_database(dimension=embedding_model.dimension)
    rag_engine = create_rag_engine(
        embedder=embedding_model,
        vector_db=vector_db
    )
    
    st.session_state.initialized = True
    return rag_engine

# Initialize document processor
@st.cache_resource
def initialize_document_processor():
    """Initialize document processor."""
    from document.processor import DocumentProcessor
    return DocumentProcessor()

# Main application
def main():
    """Main Streamlit application."""
    # Initialize components
    rag_engine = initialize_rag_engine()
    doc_processor = initialize_document_processor()
    
    # Update document count
    st.session_state.document_count = rag_engine.count_documents()
    
    # Sidebar
    st.sidebar.title("📚 RAG Document QA")
    
    # Document upload
    st.sidebar.header("Upload Documents")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a document file (PDF, TXT, DOCX)",
        type=["pdf", "txt", "md", "docx"]
    )
    
    # Upload settings
    st.sidebar.subheader("Document Settings")
    chunk_size = st.sidebar.slider(
        "Chunk Size",
        min_value=100,
        max_value=2000,
        value=1000,
        step=100,
        help="Size of text chunks in characters"
    )
    chunk_overlap = st.sidebar.slider(
        "Chunk Overlap",
        min_value=0,
        max_value=500,
        value=200,
        step=50,
        help="Overlap between chunks in characters"
    )
    
    # Search settings
    st.sidebar.header("Search Settings")
    top_k = st.sidebar.slider(
        "Results to Return",
        min_value=1,
        max_value=10,
        value=3,
        help="Number of document chunks to retrieve"
    )
    search_type = st.sidebar.selectbox(
        "Search Type",
        options=["hybrid", "semantic", "keyword"],
        index=0,
        help="Type of search to perform"
    )
    
    # Document info
    st.sidebar.header("Document Store")
    st.sidebar.metric("Documents Stored", st.session_state.document_count)
    
    if st.sidebar.button("Clear All Documents"):
        rag_engine.clear_documents()
        st.session_state.document_count = 0
        st.sidebar.success("Document store cleared!")
        st.experimental_rerun()
    
    # Process uploaded file
    if uploaded_file is not None:
        with st.sidebar.expander("Upload Status", expanded=True):
            with st.spinner('Processing document...'):
                # Save to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                try:
                    # Process document
                    doc_processor.chunk_size = chunk_size
                    doc_processor.chunk_overlap = chunk_overlap
                    
                    chunks, chunk_metadata = doc_processor.process_file(
                        tmp_file_path,
                        metadata={"filename": uploaded_file.name, "source": "UI upload"}
                    )
                    
                    if not chunks:
                        st.sidebar.error("No text could be extracted from the document.")
                    else:
                        # Add chunks to RAG engine
                        doc_ids = rag_engine.add_documents(chunks, chunk_metadata)
                        
                        # Update document count
                        st.session_state.document_count = rag_engine.count_documents()
                        
                        st.sidebar.success(f"Added {len(chunks)} document chunks!")
                except Exception as e:
                    st.sidebar.error(f"Error processing document: {str(e)}")
                finally:
                    # Clean up temporary file
                    os.unlink(tmp_file_path)
    
    # Main content
    st.title("📚 Document Query System")
    
    if st.session_state.document_count == 0:
        st.info("👈 Please upload documents using the sidebar to get started.")
        
        # Sample documents
        st.subheader("Sample Text")
        sample_text = st.text_area(
            "Or try adding some sample text directly:",
            height=200
        )
        
        if sample_text and st.button("Add Sample Text"):
            with st.spinner('Processing text...'):
                # Chunk the text
                chunks = doc_processor._chunk_text(sample_text, chunk_size, chunk_overlap)
                
                # Create metadata
                chunk_metadata = [
                    {"source": "Sample text", "chunk_id": i, "total_chunks": len(chunks)}
                    for i in range(len(chunks))
                ]
                
                # Add to RAG engine
                doc_ids = rag_engine.add_documents(chunks, chunk_metadata)
                
                # Update document count
                st.session_state.document_count = rag_engine.count_documents()
                
                st.success(f"Added {len(chunks)} text chunks!")
                st.experimental_rerun()
    else:
        # Question answering
        st.subheader("Ask a Question")
        question = st.text_input("Enter your question:")
        
        if question:
            with st.spinner('Searching for answer...'):
                try:
                    # Generate response
                    result = rag_engine.generate_response(
                        query=question,
                        top_k=top_k,
                        search_type=search_type
                    )
                    
                    # Display response
                    st.markdown("### Answer")
                    st.write(result["response"])
                    
                    # Display sources
                    st.markdown("### Sources")
                    for i, doc in enumerate(result["retrieved_documents"]):
                        with st.expander(f"Source {i+1} (Score: {doc['score']:.2f})"):
                            st.markdown(f"**Source:** {doc['metadata'].get('source', 'Unknown')}")
                            st.text(doc["text"])
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
    
    # About section
    st.sidebar.markdown("---")
    st.sidebar.info(
        "This application allows you to upload documents and ask questions about their content. "
        "The system uses embedding models for semantic search and retrieval."
    )

# Run the application
if __name__ == "__main__":
    main()
