import streamlit as st
from PyPDF2 import PdfReader
import requests
import os
import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity

# 🌍 Hugging Face API Configuration (Serverless)
my_token = os.getenv('my_repo_token')
API_URL_MISTRAL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
headers = {"Authorization": f"Bearer {my_token}"}

# 🚀 Models
EMBEDDING_MODEL = "BAAI/bge-large-en"
RERANKER_MODEL = "BAAI/bge-reranker-large"
CHUNK_SIZE = 500  # Token chunking
TOP_K = 5  # Top-k retrieval

# 🔥 Load Embedding and Reranking Models
tokenizer_embed = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
model_embed = AutoModel.from_pretrained(EMBEDDING_MODEL)

tokenizer_rerank = AutoTokenizer.from_pretrained(RERANKER_MODEL)
model_rerank = AutoModel.from_pretrained(RERANKER_MODEL)


# 🔍 **Query Mistral API for Answering**
def query_mistral(context, question):
    """Generate an answer using Mistral API."""
    prompt = f"""
You are a helpful assistant. You are given a context extracted from a document. 
Answer the question based only on the provided context. If the answer is not in the context, simply say "I don't know." Do not make up any information.

Context:
{context}

Question:
{question}

Answer:
"""
    payload = {"inputs": prompt, "parameters": {"max_length": 200, "temperature": 0.3}}
    response = requests.post(API_URL_MISTRAL, headers=headers, json=payload)

    try:
        response_json = response.json()
        if isinstance(response_json, list) and response_json:
            return response_json[0].get("generated_text", "Error: No generated text").strip()
        elif isinstance(response_json, dict) and "error" in response_json:
            return f"Error: {response_json['error']}"
        return "Error: Invalid response format"
    except Exception:
        return "Error: Failed to parse response"


# 🧠 **Compute Text Embeddings**
def get_embeddings(texts, tokenizer, model):
    """Generate embeddings using BGE-Large."""
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.pooler_output.cpu().numpy()  # Use `pooler_output` for retrieval


# 🔎 **Retrieve Most Relevant Contexts with FAISS**
def find_most_relevant_context_faiss(contexts, question):
    """Retrieve top-k contexts using FAISS similarity search."""
    all_texts = [question] + contexts
    embeddings = get_embeddings(all_texts, tokenizer_embed, model_embed)

    question_embedding = embeddings[0]
    context_embeddings = embeddings[1:]

    # Build FAISS index
    dimension = context_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(context_embeddings)

    _, indices = index.search(question_embedding.reshape(1, -1), min(TOP_K, len(context_embeddings)))

    return [contexts[idx] for idx in indices[0] if idx < len(contexts)]


# 🔎 **Retrieve Relevant Contexts with BM25**
def find_most_relevant_context_bm25(contexts, question):
    """Retrieve top-k contexts using BM25."""
    tokenized_corpus = [doc.split() for doc in contexts]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_question = question.split()
    return bm25.get_top_n(tokenized_question, contexts, n=min(TOP_K, len(contexts)))


# 📊 **Rerank Retrieved Documents with BGE-Reranker**
def rerank_results(contexts, question):
    """Rerank results using cosine similarity."""
    inputs = [f"Query: {question} Document: {context}" for context in contexts]

    # Generate embeddings
    inputs_tokenized = tokenizer_rerank(inputs, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        embeddings = model_rerank(**inputs_tokenized).pooler_output.cpu().numpy()

    query_embedding = embeddings[0]  # Query embedding
    context_embeddings = embeddings[1:]  # Context embeddings

    # Compute cosine similarity scores
    scores = cosine_similarity([query_embedding], context_embeddings)[0]

    # Sort by similarity score
    ranked_contexts = sorted(zip(contexts, scores), key=lambda x: x[1], reverse=True)

    return [context for context, _ in ranked_contexts][:TOP_K]


# 🏆 **Final Hybrid Retrieval (FAISS + BM25 + Reranking)**
def hybrid_search(contexts, question):
    """Combine FAISS, BM25, and reranking for better retrieval."""
    faiss_results = find_most_relevant_context_faiss(contexts, question)
    bm25_results = find_most_relevant_context_bm25(contexts, question)
    combined_results = list(set(faiss_results + bm25_results))

    if not combined_results:
        return ["I don't know."]
    
    return rerank_results(combined_results, question)


# 📜 **Extract Text from PDF**
def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file."""
    pdf_reader = PdfReader(pdf_file)
    pdf_arr = [pdf_reader.pages[i].extract_text() for i in range(len(pdf_reader.pages)) if pdf_reader.pages[i].extract_text()]
    return pdf_arr if pdf_arr else ["No text extracted from the PDF"]


# 🎤 **Answer Questions from Retrieved Contexts**
def answer_question_from_pdf(contexts, question):
    """Generate a final answer using the most relevant document contexts."""
    if not contexts or all(ctx.strip() == "" for ctx in contexts):
        return "I could not find relevant information in the document."
    
    combined_context = " ".join(contexts)
    return query_mistral(combined_context, question)


# 🎭 **Streamlit Chatbot UI**
st.title("📄 PDF Chatbot (Serverless)")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    pdf_arr = extract_text_from_pdf(uploaded_file)
    st.write("✅ PDF Uploaded Successfully. Start chatting!")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    question = st.text_input("Ask a question about the PDF")
    if st.button("Send") and question:
        retrieved_contexts = hybrid_search(pdf_arr, question)
        response = answer_question_from_pdf(retrieved_contexts, question)

        st.session_state.chat_history.append((question, response))

    for q, r in st.session_state.chat_history:
        st.write(f"**You:** {q}")
        st.write(f"**Bot:** {r}")

else:
    st.write("📂 Please upload a PDF file to start the chat.")
