# utils.py
import os
import time
import requests
from requests.exceptions import RequestException
from PyPDF2 import PdfReader
from config import CHUNK_SIZE, OVERLAP, API_URL_MISTRAL

# In Spaces, use HF_TOKEN from secrets
my_token = os.getenv('my_repo_token')  # Set this in Spaces secrets
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
You are a helpful assistant. Answer the question based only on the provided context. 
If the answer is not in the context, say "I don't know." Do not make up information.

Context:
{context}

Question:
{question}

Answer:
"""
    payload = {"inputs": prompt, "parameters": {"max_length": 200, "temperature": 0.3}}
    
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
    
    combined_context = " ".join(contexts)
    return query_mistral(combined_context, question)