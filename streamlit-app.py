"""
Streamlit UI for the RAG system.
"""

import os
import sys
import streamlit as st
import tempfile
import logging
from dotenv import load_dotenv

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

# Configure logging
from config import HUGGINGFACE_MODEL, get_logging_config
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
if "last_question" not in st.session_state:
    st.session_state.last_question = ""
if "use_advanced" not in st.session_state:
    st.session_state.use_advanced = False
if "advanced_top_k" not in st.session_state:
    st.session_state.advanced_top_k = 3
if "advanced_search_type" not in st.session_state:
    st.session_state.advanced_search_type = "hybrid"

# Initialize RAG engine
@st.cache_resource
def initialize_rag_engine():
    """Initialize RAG engine with appropriate LLM."""
    from embedding.model import create_embedding_model
    from storage.vector_db import create_vector_database
    from rag.engine import create_rag_engine
    from llm.model import create_llm
    
    # Create components
    embedding_model = create_embedding_model()
    
    # Use KeywordVectorDatabase if facing FAISS installation issues
    try:
        vector_db = create_vector_database(dimension=embedding_model.dimension)
    except Exception as e:
        logger.warning(f"Error creating FAISS database: {e}, falling back to keyword database")
        vector_db = create_vector_database(db_type="keyword", dimension=embedding_model.dimension)
    
    # Try to get HuggingFace API key from secrets
    api_key = None
    try:
        api_key = st.secrets.get("HUGGINGFACE_API_KEY")
    except:
        pass
    
    # Create LLM - try serverless first, fall back to local
    if api_key:
        llm = create_llm( model_type="serverless",model_name=HUGGINGFACE_MODEL, api_key=api_key)
        logger.info("Using serverless LLM with HuggingFace API")
    else:
        llm = create_llm(model_type="local")
        logger.info("Using local LLM implementation")
    
    # Create RAG engine with the LLM
    rag_engine = create_rag_engine(
        embedder=embedding_model,
        vector_db=vector_db,
        llm=llm
    )
    
    st.session_state.initialized = True
    return rag_engine

# Initialize document processor
@st.cache_resource
def initialize_document_processor():
    """Initialize document processor."""
    from document.processor import DocumentProcessor
    return DocumentProcessor(
        chunk_size=2000,
        chunk_overlap=200,
        chunking_mode="auto"
    )

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
    chunking_mode = st.sidebar.selectbox(
        "Chunking Method",
        options=["auto", "paragraph", "sentence", "character"],
        index=0,
        help="How to split your document into chunks"
    )
    
    chunk_size = st.sidebar.slider(
        "Chunk Size",
        min_value=500,
        max_value=5000,
        value=2000,  # Increased default to 2000
        step=500,
        help="Size of text chunks in characters (for character and sentence modes)"
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
        st.rerun()
    
    # Process uploaded file
    if uploaded_file is not None:
        with st.sidebar.expander("Upload Status", expanded=True):
            with st.spinner('Processing document...'):
                # Save to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                try:
                    # Update processor settings from UI
                    doc_processor.chunk_size = chunk_size
                    doc_processor.chunk_overlap = chunk_overlap
                    doc_processor.chunking_mode = chunking_mode
                    
                    # Process document
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
                        
                        # Show chunk preview
                        with st.sidebar.expander("Document Preview", expanded=False):
                            st.write(f"First chunk ({len(chunks[0])} chars):")
                            st.text_area("", chunks[0][:300] + "...", height=100, key="preview_chunk")
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
            height=200,
            key="sample_text_input"
        )
        
        if sample_text and st.button("Add Sample Text"):
            with st.spinner('Processing text...'):
                # Update processor settings from UI
                doc_processor.chunk_size = chunk_size
                doc_processor.chunk_overlap = chunk_overlap
                doc_processor.chunking_mode = chunking_mode
                
                # Chunk the text using the document processor
                chunks = doc_processor._chunk_by_paragraphs(sample_text) if chunking_mode == "paragraph" else \
                         doc_processor._chunk_by_sentences(sample_text, chunk_size, chunk_overlap) if chunking_mode == "sentence" else \
                         doc_processor._chunk_text(sample_text, chunk_size, chunk_overlap)
                
                # Create metadata
                chunk_metadata = [
                    {"source": "Sample text", "chunk_id": i, "total_chunks": len(chunks), 
                     "chunk_position": f"{i+1}/{len(chunks)}", "chunk_size_chars": len(chunk),
                     "chunk_size_words": len(chunk.split())}
                    for i, chunk in enumerate(chunks)
                ]
                
                # Add to RAG engine
                doc_ids = rag_engine.add_documents(chunks, chunk_metadata)
                
                # Update document count
                st.session_state.document_count = rag_engine.count_documents()
                
                st.success(f"Added {len(chunks)} text chunks!")
                st.rerun()
    else:
        # Question answering
        st.subheader("Ask a Question")
        
        # Add query presets
        query_type = st.radio(
            "Question Type",
            options=["Custom Question", "Summarize Document", "Technical Analysis", "Compare Content"],
            horizontal=True,
            key="query_type_radio"
        )
        
        if query_type == "Custom Question":
            question = st.text_input("Enter your question:", key="question_input")
        elif query_type == "Summarize Document":
            question = "Summarize the key points of this document."
        elif query_type == "Technical Analysis":
            question = "Analyze the technical aspects of this content. What technologies, methods, or frameworks are mentioned?"
        else:  # Compare Content
            question = "Compare and contrast the different sections or concepts presented in this document."
            
        # Advanced options using session state
        with st.expander("Advanced Options", expanded=False):
            st.session_state.advanced_top_k = st.slider(
                "Number of chunks to retrieve",
                min_value=1,
                max_value=10,
                value=st.session_state.advanced_top_k,
                key="advanced_top_k_slider"
            )
            
            st.session_state.advanced_search_type = st.selectbox(
                "Search Method",
                options=["hybrid", "semantic", "keyword"],
                index=["hybrid", "semantic", "keyword"].index(st.session_state.advanced_search_type),
                key="advanced_search_type_select"
            )
            
            st.session_state.use_advanced = st.checkbox(
                "Use advanced settings", 
                value=st.session_state.use_advanced,
                key="use_advanced_checkbox"
            )
        
        # Use advanced settings if enabled
        if st.session_state.use_advanced:
            top_k = st.session_state.advanced_top_k
            search_type = st.session_state.advanced_search_type
            st.caption("Using advanced search settings")
        
        # Debug mode
        debug_mode = st.checkbox("Debug Mode", value=False, key="debug_mode")
        
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
                    
                    # Add metadata about the response
                    if "template_used" in result:
                        st.caption(f"Using template: {result['template_used']}")
                    
                    # Show debug information if enabled
                    if debug_mode:
                        with st.expander("Debug Information", expanded=True):
                            st.subheader("Retrieved Documents")
                            st.json(result["retrieved_documents"])
                            
                            # Show template being used
                            st.subheader("Prompt Template")
                            if "template_used" in result:
                                try:
                                    from config import (
                                        DEFAULT_PROMPT_TEMPLATE,
                                        ENHANCED_PROMPT_TEMPLATE,
                                        DOCUMENT_SUMMARY_TEMPLATE,
                                        DOCUMENT_COMPARE_TEMPLATE,
                                        TECHNICAL_DOCUMENT_TEMPLATE
                                    )
                                    templates = {
                                        "default": DEFAULT_PROMPT_TEMPLATE,
                                        "enhanced": ENHANCED_PROMPT_TEMPLATE,
                                        "summary": DOCUMENT_SUMMARY_TEMPLATE,
                                        "compare": DOCUMENT_COMPARE_TEMPLATE,
                                        "technical": TECHNICAL_DOCUMENT_TEMPLATE
                                    }
                                    template = templates.get(result["template_used"], "Unknown template")
                                    st.text_area("Template", template, height=200, key="debug_template")
                                except ImportError:
                                    st.warning("Could not load template definitions")
                    
                    # Display sources with tabs
                    st.markdown("### Sources")
                    
                    if len(result["retrieved_documents"]) > 0:
                        tabs = st.tabs([f"Source {i+1}" for i in range(len(result["retrieved_documents"]))])
                        
                        for i, (tab, doc) in enumerate(zip(tabs, result["retrieved_documents"])):
                            with tab:
                                # Main content
                                st.text_area(
                                    "Content",
                                    doc["text"],
                                    height=150,
                                    key=f"source_text_{i}"  # Add unique key
                                )
                                
                                # Metadata section
                                meta = doc["metadata"]
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown("**Source Info**")
                                    st.text(f"Source: {meta.get('source', 'Unknown')}")
                                    if "filename" in meta:
                                        st.text(f"File: {meta.get('filename')}")
                                    if "chunk_position" in meta:
                                        st.text(f"Position: {meta.get('chunk_position')}")
                                    if "content_type" in meta:
                                        st.text(f"Type: {meta.get('content_type')}")
                                
                                with col2:
                                    st.markdown("**Content Stats**")
                                    if "word_count" in meta:
                                        st.text(f"Words: {meta.get('word_count')}")
                                    if "potential_title" in meta:
                                        st.text(f"Title: {meta.get('potential_title')}")
                                    if "language" in meta:
                                        st.text(f"Language: {meta.get('language')}")
                                    st.text(f"Relevance: {doc.get('score', 0):.2f}")
                    else:
                        st.info("No source documents were retrieved.")
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
                    if debug_mode:
                        st.exception(e)  # Show full traceback in debug mode
                    
        # Add feedback mechanism
        if "last_question" in st.session_state and st.session_state.last_question == question and question:
            feedback = st.radio(
                "Was this answer helpful?",
                options=["", "Yes", "Partially", "No"],
                horizontal=True,
                index=0,
                key="feedback_radio"
            )
            
            if feedback in ["Partially", "No"]:
                feedback_text = st.text_area(
                    "How can we improve this answer?", 
                    height=100, 
                    key="feedback_text"
                )
                if st.button("Submit Feedback", key="submit_feedback"):
                    st.success("Thank you for your feedback!")
        
        # Update last question in session state
        if question:
            st.session_state.last_question = question
    
    # About section
    st.sidebar.markdown("---")
    st.sidebar.info(
        "This application allows you to upload documents and ask questions about their content. "
        "The system uses embedding models for semantic search and retrieval."
    )

# Run the application
if __name__ == "__main__":
    main()