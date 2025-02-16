

import streamlit as st
from PyPDF2 import PdfReader
import requests
import os
import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util


my_token = os.getenv('my_repo_token')
# Use Mistral API for serverless architecture
API_URL_MISTRAL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
headers = {"Authorization": f"Bearer {my_token}"}

# Initialize Retrieval Models
bm25_model = None
dense_retrieval_model = SentenceTransformer("sentence-transformers/msmarco-MiniLM-L6-cos-v5")
reranker_tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
reranker_model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")

# 🔹 Query Mistral API for Response Generation
def query_mistral(prompt):
    payload = {"inputs": prompt, "parameters": {"max_length": 200, "temperature": 0.7}}
    response = requests.post(API_URL_MISTRAL, headers=headers, json=payload)
    try:
        response_json = response.json()
        if isinstance(response_json, list) and response_json:
            generated_text = response_json[0].get('generated_text', "Error: No generated text")
            return generated_text.replace("Based on this content:", "").strip()
        elif isinstance(response_json, dict) and "error" in response_json:
            return f"Error: {response_json['error']}"
        return "Error: Invalid response format"
    except:
        return "Error: Failed to parse response"

# 🔹 Generate a Response with Safe Prompting
def generate_response(context, question):
    return query_mistral(
        f"""You are an AI assistant answering questions based on a document.
        Use the provided document **only**. If the document does not contain relevant information, say 'I don’t know.'
        \n\n Document: {context} \n\nQuestion: {question}"""
    )

# 🔹 Get Dense Embeddings for FAISS Search
def get_embeddings(texts, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    
    return embeddings

# 🔹 FAISS Search (Semantic Search)
def find_most_relevant_context_faiss(contexts, question):
    all_texts = [question] + contexts
    embeddings = get_embeddings(all_texts)
    
    question_embedding = embeddings[0]
    context_embeddings = embeddings[1:]
    
    dimension = context_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(context_embeddings)
    
    _, indices = index.search(question_embedding.reshape(1, -1), min(3, len(context_embeddings)))
    return [contexts[idx] for idx in indices[0] if idx < len(contexts)]

# 🔹 BM25 Search (Keyword-Based)
def find_most_relevant_context_bm25(contexts, question):
    global bm25_model
    tokenized_corpus = [doc.split() for doc in contexts]
    
    if bm25_model is None:
        bm25_model = BM25Okapi(tokenized_corpus)
    
    tokenized_question = question.split()
    top_docs = bm25_model.get_top_n(tokenized_question, contexts, n=min(3, len(contexts)))
    return top_docs

# 🔹 Rerank Retrieved Passages with Cross-Encoder
def rerank(query, passages):
    scores = []
    for passage in passages:
        inputs = reranker_tokenizer(query, passage, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            score = reranker_model(**inputs).logits.squeeze().item()
        scores.append((passage, score))
    return sorted(scores, key=lambda x: x[1], reverse=True)[0][0]

# 🔹 Verify Answer with a Second Model
def verify_answer(question, generated_answer):
    verifier_prompt = f"""Verify if the following answer is grounded in the given question.
    If it includes hallucinated details, respond with 'Incorrect'. Otherwise, say 'Correct'.
    
    Question: {question}
    Answer: {generated_answer}"""
    
    verification_response = query_mistral(verifier_prompt)
    return verification_response.lower() == "correct"

# 🔹 Extract Text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    pdf_arr = [pdf_reader.pages[page_num].extract_text() for page_num in range(len(pdf_reader.pages)) if pdf_reader.pages[page_num].extract_text()]
    return pdf_arr if pdf_arr else ["No text extracted from the PDF"]

# 🔹 Answer Question with RAG and Hallucination Prevention
def answer_question_from_pdf(pdf_text, question):
    faiss_results = find_most_relevant_context_faiss(pdf_text, question)
    bm25_results = find_most_relevant_context_bm25(pdf_text, question)
    combined_results = list(set(faiss_results + bm25_results))
    
    if not combined_results:
        return "I don’t know. The document does not contain relevant information."

    top_context = rerank(question, combined_results)
    response = generate_response(top_context, question)

    # Apply verification
    if not verify_answer(question, response):
        return "I'm not confident in my answer. Please check the document directly."

    return response

# 🔹 Streamlit UI
st.title("Hallucination-Free PDF Chatbot")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    pdf_arr = extract_text_from_pdf(uploaded_file)
    st.write("PDF Uploaded Successfully. Start chatting!")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    question = st.text_input("Ask a question about the PDF")
    if st.button("Send") and question:
        response = answer_question_from_pdf(pdf_arr, question)
        st.session_state.chat_history.append((question, response))
    
    for q, r in st.session_state.chat_history:
        st.write(f"**You:** {q}")
        st.write(f"**Bot:** {r}")
else:
    st.write("Please upload a PDF file to start the chat.")
    

