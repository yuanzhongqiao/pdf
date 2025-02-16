

import streamlit as st
from PyPDF2 import PdfReader
import requests
import os
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from rank_bm25 import BM25Okapi
import json
from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast, DPRQuestionEncoder, DPRQuestionEncoderTokenizerFast

my_token = os.getenv('my_repo_token')
# Use Mistral API for serverless architecture
API_URL_MISTRAL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
headers = {"Authorization": f"Bearer {my_token}"}

# Query Mistral API
def query_mistral(payload):
    response = requests.post(API_URL_MISTRAL, headers=headers, json=payload)
    try:
        response_json = response.json()
        if isinstance(response_json, list) and response_json:
            return response_json[0].get('generated_text', "Error: No generated text").replace("Based on this content:", "").strip()
        elif isinstance(response_json, dict) and "error" in response_json:
            return f"Error: {response_json['error']}"
        return "Error: Invalid response format"
    except json.JSONDecodeError:
        return "Error: Failed to parse response"

# Generate response from Mistral
def generate_response(prompt):
    payload = {"inputs": prompt, "parameters": {"max_length": 200, "temperature": 0.7}}
    return query_mistral(payload)

# Dense Retrieval - FAISS Embeddings
def get_embeddings(texts, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    texts = [str(text) for text in texts if text]  # Ensure valid text input
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)
    
    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return embeddings

# Sparse Retrieval - BM25
def find_most_relevant_context_bm25(contexts, question):
    tokenized_corpus = [doc.split() for doc in contexts]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_question = question.split()
    return bm25.get_top_n(tokenized_question, contexts, n=min(3, len(contexts)))

# Dense Retrieval - FAISS
def find_most_relevant_context_faiss(contexts, question, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    all_texts = [question] + contexts
    all_texts = [str(text) for text in all_texts if text]  # Ensure valid input
    
    if not all_texts:
        return []

    embeddings = get_embeddings(all_texts, model_name=model_name)
    question_embedding = embeddings[0]
    context_embeddings = embeddings[1:]

    dimension = context_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(context_embeddings)

    _, indices = index.search(question_embedding.reshape(1, -1), min(3, len(context_embeddings)))
    return [contexts[idx] for idx in indices[0] if idx < len(contexts)]

# **ColBERT Retrieval - Contextualized Word-Level Matching**
def find_most_relevant_context_colbert(contexts, question):
    # Load DPR question and context encoders (ColBERT-like approach)
    question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    question_tokenizer = DPRQuestionEncoderTokenizerFast.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    context_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

    # Encode question
    question_inputs = question_tokenizer(question, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        question_embedding = question_encoder(**question_inputs).pooler_output

    # Encode each context
    context_embeddings = []
    for context in contexts:
        context_inputs = context_tokenizer(context, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            context_embeddings.append(context_encoder(**context_inputs).pooler_output)
    
    context_embeddings = torch.stack(context_embeddings)

    # Compute similarity scores
    scores = torch.matmul(question_embedding, context_embeddings.T).squeeze(0)
    top_k_indices = scores.topk(min(3, len(contexts))).indices.tolist()

    return [contexts[idx] for idx in top_k_indices]

# Hybrid Retrieval - FAISS + BM25 + ColBERT
def hybrid_search(contexts, question):
    faiss_results = find_most_relevant_context_faiss(contexts, question)
    bm25_results = find_most_relevant_context_bm25(contexts, question)
    colbert_results = find_most_relevant_context_colbert(contexts, question)
    
    # Combine results, ensuring uniqueness
    combined_results = list(set(faiss_results + bm25_results + colbert_results))
    return combined_results if combined_results else ["No relevant context found."]

# Generate answer based on retrieved context
def answer_question_from_pdf(pdf_text, question):
    return generate_response(f"{pdf_text} The Question is: {question} Provide the answer with max length of about 100 words.")

# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    pdf_arr = [pdf_reader.pages[page_num].extract_text() for page_num in range(len(pdf_reader.pages)) if pdf_reader.pages[page_num].extract_text()]
    return pdf_arr if pdf_arr else ["No text extracted from the PDF"]

# Streamlit chatbot UI
st.title("PDF Chatbot (Serverless) - Hybrid RAG (FAISS + BM25 + ColBERT)")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    pdf_arr = extract_text_from_pdf(uploaded_file)
    st.write("📄 **PDF Uploaded Successfully. Start chatting!**")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    question = st.text_input("Ask a question about the PDF")
    if st.button("Send") and question:
        combined_results = hybrid_search(pdf_arr, question)  # Hybrid Retrieval
        
        # Generate response using the best retrieved contexts
        response = answer_question_from_pdf(" ".join(combined_results), question)

        # Store the conversation history
        st.session_state.chat_history.append((question, response))

    # Display chat history
    for q, r in st.session_state.chat_history:
        st.write(f"**You:** {q}")
        st.write(f"**Bot:** {r}")
else:
    st.write("📥 **Please upload a PDF file to start the chat.**")
