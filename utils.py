# utils.py
import os
import time
import requests
from requests.exceptions import RequestException
from PyPDF2 import PdfReader
import logging
from config import CHUNK_SIZE, OVERLAP, API_URL_MISTRAL

# Logging setup
logger = logging.getLogger(__name__)

# Use HF_TOKEN from Spaces secrets
my_token = os.getenv('HF_TOKEN')
headers = {"Authorization": f"Bearer {my_token}"}

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    tokens = text.split()
    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk = " ".join(tokens[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    pdf_arr = [page.extract_text() for page in pdf_reader.pages if page.extract_text()]
    if not pdf_arr:
        return None
    
    chunks = []
    for page_text in pdf_arr:
        chunks.extend(chunk_text(page_text))
    return chunks if chunks else ["No text extracted from the PDF"]

def query_mistral(context, question, retries=3):
    prompt = f"""
You are a precise and factual assistant. Your sole task is to answer the question based *only* on the provided context. 
Do not infer, extrapolate, or add information beyond what is explicitly stated in the context. 
If the context does not contain the answer, respond with "I don't know." 
Do not use external knowledge or make assumptions.

Context:
{context}

Question:
{question}

Answer:
"""
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_length": 200,
            "temperature": 0.1,
            "top_p": 0.9,
            "repetition_penalty": 1.2
        }
    }
    
    for attempt in range(retries):
        try:
            response = requests.post(API_URL_MISTRAL, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()[0].get("generated_text", "Error").strip()
        except RequestException as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return f"Sorry, I couldn't get an answer due to an API issue: {str(e)}"

def answer_question_from_pdf(contexts, question):
    if not contexts or all(ctx.strip() == "" for ctx in contexts):
        return "I couldn't find relevant information in the document."
    
    combined_context = " ".join(contexts[:2])  # Use top 2 contexts
    response = query_mistral(combined_context, question)
    
    # Hallucination check
    response_words = set(response.lower().split())
    context_words = set(combined_context.lower().split())
    key_terms = set(question.lower().split())
    
    unexplained_terms = response_words - (context_words | key_terms)
    if unexplained_terms and "I don't know" not in response:
        logger.warning(f"Possible hallucination detected: {unexplained_terms}")
        return "I don't know."
    
    return response