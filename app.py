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

# Use Mistral API for serverless architecture
my_token = os.getenv('my_repo_token')
API_URL_MISTRAL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
headers = {"Authorization": f"Bearer {my_token}"}

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

def generate_response(prompt):
    prompt += "\nIf the answer is not in the provided text, say 'I don't know'."
    payload = {"inputs": prompt, "parameters": {"max_length": 200, "temperature": 0.3}}
    return query_mistral(payload)

def get_embeddings(texts, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    """Compute dense vector embeddings with proper text format handling."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Ensure input is a list of non-empty strings
    texts = [str(text) if text else " " for text in texts]  

    # Tokenize correctly
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

    with torch.no_grad():
        outputs = model(**inputs)

    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return embeddings

def find_most_relevant_context_faiss(contexts, question, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    """Find most relevant context using FAISS for dense retrieval."""
    all_texts = [question] + contexts
    embeddings = get_embeddings(all_texts, model_name=model_name)

    if len(embeddings) < 2:
        return []

    question_embedding = embeddings[0]
    context_embeddings = embeddings[1:]

    dimension = context_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(context_embeddings)

    _, indices = index.search(question_embedding.reshape(1, -1), min(3, len(context_embeddings)))
    return [contexts[idx] for idx in indices[0] if idx < len(contexts)]

def find_most_relevant_context_bm25(contexts, question):
    """Find relevant context using BM25 for sparse retrieval."""
    tokenized_corpus = [doc.split() for doc in contexts if doc]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_question = question.split()
    top_docs = bm25.get_top_n(tokenized_question, contexts, n=min(3, len(contexts)))
    return top_docs

def rerank_results(contexts, question):
    """Rerank retrieved results based on relevance."""
    scores = []
    for context in contexts:
        response = generate_response(f"Does this text answer the question: '{question}'? Text: {context}")
        if "yes" in response.lower():
            scores.append((context, 1))
        else:
            scores.append((context, 0))
    ranked_contexts = [context for context, score in scores if score > 0]
    return ranked_contexts if ranked_contexts else ["I don't know."]

def is_response_confident(response):
    """Check for uncertainty in AI response."""
    low_confidence_phrases = [
        "i'm not sure", "i think", "possibly", "maybe", "it seems", "likely",
        "i guess", "i assume", "it's possible", "as far as i know"
    ]
    return not any(phrase in response.lower() for phrase in low_confidence_phrases)

def validate_response(prompt):
    """Ensure consistent answers by checking multiple responses."""
    response1 = generate_response(prompt)
    response2 = generate_response(prompt)
    if response1.strip().lower() != response2.strip().lower():
        return "I am not confident about the answer. Please refer to the document."
    return response1

def answer_question_from_pdf(pdf_text, question):
    """Answer the question using retrieved PDF context while preventing hallucination."""
    if not pdf_text.strip():
        return "I could not find relevant information in the document."
    
    prompt = f"{pdf_text} The Question is: {question} Provide the answer with max length of about 100 words."
    response = validate_response(prompt)

    if not is_response_confident(response):
        return "I'm not sure about this. Please refer to the original document."

    return response

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file."""
    pdf_reader = PdfReader(pdf_file)
    pdf_arr = [pdf_reader.pages[page_num].extract_text() for page_num in range(len(pdf_reader.pages)) if pdf_reader.pages[page_num].extract_text()]
    return pdf_arr if pdf_arr else ["No text extracted from the PDF"]

def hybrid_search(contexts, question):
    """Combine FAISS and BM25 for hybrid retrieval."""
    faiss_results = find_most_relevant_context_faiss(contexts, question)
    bm25_results = find_most_relevant_context_bm25(contexts, question)
    combined_results = list(set(faiss_results + bm25_results))
    
    if not combined_results:
        return ["I don't know."]
    
    return rerank_results(combined_results, question)

# Streamlit chatbot UI
st.title("PDF Chatbot (Serverless with Hybrid Retrieval)")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    pdf_arr = extract_text_from_pdf(uploaded_file)
    st.write("PDF Uploaded Successfully. Start chatting!")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    question = st.text_input("Ask a question about the PDF")
    
    if st.button("Send") and question:
        combined_results = hybrid_search(pdf_arr, question)
        response = answer_question_from_pdf(" ".join(combined_results), question) if combined_results else "No relevant context found."
        st.session_state.chat_history.append((question, response))
    
    for q, r in st.session_state.chat_history:
        st.write(f"**You:** {q}")
        st.write(f"**Bot:** {r}")
else:
    st.write("Please upload a PDF file to start the chat.")
