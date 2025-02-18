import streamlit as st
from PyPDF2 import PdfReader
import requests
import os
import faiss
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# Hugging Face API settings
my_token = os.getenv('my_repo_token')
API_URL_MISTRAL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
headers = {"Authorization": f"Bearer {my_token}"}

# Model settings
EMBEDDING_MODEL = "BAAI/bge-large-en"
RERANKER_MODEL = "BAAI/bge-reranker-large"
CHUNK_SIZE = 500  # Max 500 tokens per chunk
TOP_K = 5  # Retrieve top 5 results

# Load models
tokenizer_emb = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
model_emb = AutoModel.from_pretrained(EMBEDDING_MODEL)

tokenizer_rerank = AutoTokenizer.from_pretrained(RERANKER_MODEL)
model_rerank = AutoModel.from_pretrained(RERANKER_MODEL)


# ====== HELPER FUNCTIONS ======

# Query Mistral API (Serverless)
def query_mistral(payload):
    response = requests.post(API_URL_MISTRAL, headers=headers, json=payload)
    try:
        response_json = response.json()
        if isinstance(response_json, list) and response_json:
            return response_json[0].get('generated_text', "Error: No generated text").replace("Based on this content:", "").strip()
        elif isinstance(response_json, dict) and "error" in response_json:
            return f"Error: {response_json['error']}"
        return "Error: Invalid response format"
    except Exception:
        return "Error: Failed to parse response"


# Generate response from Mistral
def generate_response(prompt):
    payload = {"inputs": prompt, "parameters": {"max_length": 200, "temperature": 0.3}}
    return query_mistral(payload)


# Split text into chunks of 500 tokens
def chunk_text(text, chunk_size=CHUNK_SIZE):
    tokens = tokenizer_emb.tokenize(text)
    chunks = [" ".join(tokens[i: i + chunk_size]) for i in range(0, len(tokens), chunk_size)]
    return chunks


# Get embeddings for texts
def get_embeddings(texts, model_name=EMBEDDING_MODEL):
    inputs = tokenizer_emb(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model_emb(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return embeddings


# Find most relevant context using FAISS
def find_most_relevant_context_faiss(contexts, question):
    all_texts = [question] + contexts
    embeddings = get_embeddings(all_texts)
    question_embedding = embeddings[0]
    context_embeddings = embeddings[1:]

    # FAISS Index
    dimension = context_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(context_embeddings)

    # Search Top-K
    _, indices = index.search(question_embedding.reshape(1, -1), TOP_K)
    return [contexts[idx] for idx in indices[0] if idx < len(contexts)]


# Find most relevant context using BM25
def find_most_relevant_context_bm25(contexts, question):
    tokenized_corpus = [doc.split() for doc in contexts]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_question = question.split()
    top_docs = bm25.get_top_n(tokenized_question, contexts, n=TOP_K)
    return top_docs


from sklearn.metrics.pairwise import cosine_similarity

def rerank_results(contexts, question):
    """Reranks results using cosine similarity of embeddings."""
    inputs = [f"Query: {question} Document: {context}" for context in contexts]
    
    # Get embeddings
    inputs_tokenized = tokenizer_rerank(inputs, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        embeddings = model_rerank(**inputs_tokenized).pooler_output.cpu().numpy()  # Use embeddings instead of logits

    query_embedding = embeddings[0]  # Query embedding
    context_embeddings = embeddings[1:]  # Context embeddings
    
    # Compute cosine similarity scores
    scores = cosine_similarity([query_embedding], context_embeddings)[0]
    
    # Sort by similarity score
    ranked_contexts = sorted(zip(contexts, scores), key=lambda x: x[1], reverse=True)
    
    return [context for context, _ in ranked_contexts][:TOP_K]



# Hybrid search (FAISS + BM25 + ColBERT-style reranking)
def hybrid_search(contexts, question):
    faiss_results = find_most_relevant_context_faiss(contexts, question)
    bm25_results = find_most_relevant_context_bm25(contexts, question)
    combined_results = list(set(faiss_results + bm25_results))
    if not combined_results:
        return ["I don't know."]
    return rerank_results(combined_results, question)


# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = " ".join([pdf_reader.pages[i].extract_text() or "" for i in range(len(pdf_reader.pages))])
    return chunk_text(text) if text else ["No text extracted from the PDF"]


# Answer questions from PDF
def answer_question_from_pdf(pdf_text, question):
    if not pdf_text.strip():
        return "I could not find relevant information in the document."
    
    prompt = f"{pdf_text} The Question is: {question} Provide the answer with max length of about 100 words."
    response = generate_response(prompt)
    return response if response else "I'm not sure. Please refer to the document."


# ====== STREAMLIT CHATBOT UI ======
st.title("PDF Chatbot (Serverless)")

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
