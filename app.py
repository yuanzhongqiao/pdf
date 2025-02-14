import streamlit as st
from PyPDF2 import PdfReader
import pytesseract
from PIL import Image
import fitz
import io
import requests
import os
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from rank_bm25 import BM25Okapi
import json

my_token = os.getenv('my_repo_token')

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
    
    _, indices = index.search(question_embedding.reshape(1, -1), 3)  # Retrieve top-3
    
    return [contexts[idx] for idx in indices[0]]

def find_most_relevant_context_bm25(contexts, question):
    tokenized_corpus = [doc.split() for doc in contexts]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_question = question.split()
    top_docs = bm25.get_top_n(tokenized_question, contexts, n=3)
    return top_docs

API_URL_LLMA = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"
headers = {"Authorization": f"Bearer {my_token}"}

def query(payload):
    response = requests.post(API_URL_LLMA, headers=headers, json=payload, stream=True)
    collected_text = ""
    for chunk in response.iter_lines():
        if chunk:
            try:
                chunk_data = json.loads(chunk.decode())
                if "generated_text" in chunk_data:
                    collected_text += chunk_data["generated_text"] + " "
                    yield collected_text.strip()
            except json.JSONDecodeError:
                continue

def answer_question_from_pdf(pdf_text, question):
    return query({"inputs": f"Based on this content: {pdf_text} The Question is: {question} Provide the answer with max length of about 100 words."})

def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    pdf_arr = [pdf_reader.pages[page_num].extract_text() for page_num in range(len(pdf_reader.pages))]
    return pdf_arr

st.title("PDF Explorer")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    pdf_arr = extract_text_from_pdf(uploaded_file)
    st.write("PDF Uploaded Successfully.")
    
    question = st.text_input("Ask a question about the PDF")
    faiss_results = find_most_relevant_context_faiss(pdf_arr, question)
    bm25_results = find_most_relevant_context_bm25(pdf_arr, question)
    combined_results = list(set(faiss_results + bm25_results))  # Merge FAISS & BM25 results
    
    if st.button("Get Answer"):
        if question:
            response_container = st.empty()
            full_response = ""
            for chunk in answer_question_from_pdf(" ".join(combined_results), question):
                full_response = chunk  # Keep updating the full response
                response_container.write(full_response)  # Display progressively
        else:
            st.write("Please enter a question.")
else:
    st.write("Please upload a PDF file.")
