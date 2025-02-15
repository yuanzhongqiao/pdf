
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
import time
from transformers import pipeline

my_token = os.getenv('my_repo_token')
# Use Mistral API for serverless architecture
API_URL_MISTRAL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
headers = {"Authorization": f"Bearer {'my_token'}"}
def query_mistral(payload):
    response = requests.post(API_URL_MISTRAL, headers=headers, json=payload)
    try:
        response_json = response.json()
        if isinstance(response_json, list) and response_json:
            return response_json[0].get('generated_text', "Error: No generated text")
        elif isinstance(response_json, dict) and "error" in response_json:
            return f"Error: {response_json['error']}"
        return "Error: Invalid response format"
    except json.JSONDecodeError:
        return "Error: Failed to parse response"

def generate_response(prompt):
    payload = {"inputs": prompt, "parameters": {"max_length": 200, "temperature": 0.7}}
    return query_mistral(payload)

def get_embeddings(texts, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    
    return embeddings

def find_most_relevant_context_faiss(contexts, question, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    all_texts = [question] + contexts
    embeddings = get_embeddings(all_texts, model_name=model_name)
    
    question_embedding = embeddings[0]
    context_embeddings = embeddings[1:]
    
    dimension = context_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(context_embeddings)
    
    _, indices = index.search(question_embedding.reshape(1, -1), min(3, len(context_embeddings)))  # Retrieve top-3
    
    return [contexts[idx] for idx in indices[0] if idx < len(contexts)]

def find_most_relevant_context_bm25(contexts, question):
    tokenized_corpus = [doc.split() for doc in contexts]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_question = question.split()
    top_docs = bm25.get_top_n(tokenized_question, contexts, n=min(3, len(contexts)))
    return top_docs

def answer_question_from_pdf(pdf_text, question):
    return generate_response(f"Based on this content: {pdf_text} The Question is: {question} Provide the answer with max length of about 100 words.")

def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    pdf_arr = [pdf_reader.pages[page_num].extract_text() for page_num in range(len(pdf_reader.pages)) if pdf_reader.pages[page_num].extract_text()]
    return pdf_arr if pdf_arr else ["No text extracted from the PDF"]

# Streamlit chatbot UI
st.title("PDF Chatbot (Serverless)")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    pdf_arr = extract_text_from_pdf(uploaded_file)
    st.write("PDF Uploaded Successfully. Start chatting!")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    question = st.text_input("Ask a question about the PDF")
    if st.button("Send") and question:
        faiss_results = find_most_relevant_context_faiss(pdf_arr, question)
        bm25_results = find_most_relevant_context_bm25(pdf_arr, question)
        combined_results = list(set(faiss_results + bm25_results))  # Merge FAISS & BM25 results
        response = answer_question_from_pdf(" ".join(combined_results), question) if combined_results else "No relevant context found."
        
        st.session_state.chat_history.append((question, response))
    
    for q, r in st.session_state.chat_history:
        st.write(f"**You:** {q}")
        st.write(f"**Bot:** {r}")
else:
    st.write("Please upload a PDF file to start the chat.")
