# app.py
import streamlit as st
from transformers import AutoTokenizer, AutoModel
import torch
import logging
from config import EMBEDDING_MODEL, RERANKER_MODEL
from retrieval import hybrid_search, get_embeddings
from utils import extract_text_from_pdf, answer_question_from_pdf

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load models (cached by Hugging Face Spaces runtime)
@st.cache_resource
def load_models():
    tokenizer_embed = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    model_embed = AutoModel.from_pretrained(EMBEDDING_MODEL)
    tokenizer_rerank = AutoTokenizer.from_pretrained(RERANKER_MODEL)
    model_rerank = AutoModel.from_pretrained(RERANKER_MODEL)
    return tokenizer_embed, model_embed, tokenizer_rerank, model_rerank

tokenizer_embed, model_embed, tokenizer_rerank, model_rerank = load_models()

# Streamlit UI
st.title("📄 PDF Chatbot (Serverless)")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    if "pdf_arr" not in st.session_state or st.session_state.uploaded_file_name != uploaded_file.name:
        with st.spinner("Processing PDF..."):
            pdf_arr = extract_text_from_pdf(uploaded_file)
            if pdf_arr is None:
                st.warning("No readable text found in the PDF.")
            else:
                st.session_state.pdf_arr = pdf_arr
                embeddings = get_embeddings(pdf_arr, tokenizer_embed, model_embed)
                if embeddings.size == 0:
                    st.error("Failed to generate embeddings from the PDF. Please try another file.")
                else:
                    st.session_state.embeddings = embeddings
                    st.session_state.uploaded_file_name = uploaded_file.name
                    st.success("✅ PDF Uploaded Successfully. Start chatting!")
    
    if "pdf_arr" in st.session_state and "embeddings" in st.session_state:
        # Chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Question input
        question = st.text_input("Ask a question about the PDF")
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Send") and question:
                with st.spinner("Thinking..."):
                    retrieved_contexts = hybrid_search(
                        st.session_state.pdf_arr, question, st.session_state.embeddings,
                        tokenizer_embed, model_embed, tokenizer_rerank, model_rerank
                    )
                    response = answer_question_from_pdf(retrieved_contexts, question)
                    st.session_state.chat_history.append((question, response))
                    logger.info(f"Question: {question}, Retrieved contexts: {len(retrieved_contexts)}")
        
        with col2:
            if st.button("Clear Chat"):
                st.session_state.chat_history = []
                st.success("Chat history cleared.")

        # Display chat history with feedback
        for i, (q, r) in enumerate(st.session_state.chat_history):
            st.write(f"**You:** {q}")
            if "Error" in r or "Sorry" in r:
                st.error(r)
            else:
                st.write(f"**Bot:** {r}")
            with st.expander("See retrieved contexts"):
                st.write(retrieved_contexts)
            if st.button("Flag as Incorrect", key=f"flag_{i}"):
                logger.info(f"Flagged response: Q: {q}, R: {r}")
                st.warning("Thanks for your feedback! We'll review this response.")

else:
    st.write("📂 Please upload a PDF file to start the chat.")