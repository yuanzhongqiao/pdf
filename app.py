import streamlit as st
from PyPDF2 import PdfReader
import pytesseract
from PIL import Image
import fitz
import io
# from transformers import pipeline
import requests
import os

import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
my_token = os.getenv('my_repo_token')
# Function to get embeddings using a pre-trained model
def get_embeddings(texts, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    
    return embeddings

# Function to find the most relevant context using FAISS
def find_most_relevant_context(contexts, question, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    # Get embeddings for contexts and question
    all_texts = [question] + contexts
    embeddings = get_embeddings(all_texts, model_name=model_name)
    
    # Separate the question embedding and context embeddings
    question_embedding = embeddings[0]
    context_embeddings = embeddings[1:]
    
    # Create a FAISS index and add context embeddings
    dimension = context_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(context_embeddings)
    
    # Search for the nearest neighbor to the question embedding
    _, indices = index.search(question_embedding.reshape(1, -1), 1)
    
    # Get the most relevant context
    most_relevant_index = indices[0][0]
    return contexts[most_relevant_index]










API_URL = "https://api-inference.huggingface.co/models/google/gemma-7b"
API_URL_2 = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-v0.1"
API_URL_LLMA = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"
headers = {"Authorization": f"Bearer {my_token}"}
# pipe = pipeline("text-generation", model="mistralai/Mixtral-8x7B-v0.1", token = my_token)

def query(payload):
	response = requests.post(API_URL_LLMA, headers=headers, json=payload)
    
	return response.json()
    # return pipe(payload)
	



# Mock function for answering questions from the PDF
# Replace this with your actual backend function
def answer_question_from_pdf(pdf_text, question):
    # This function should return the answer to the question based on the PDF content
    # Here we just return a mock response
 
    answer = query(   {"inputs": "Based on this content: " + pdf_text+" The Question is: "+ question + " Provide the answer with max lenghth of about 1500",})
    answer = answer[0]["generated_text"]
    answer = answer[answer.find("Answer")+6:]
    return answer
# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    pdf_text = ""
    pdf_arr = []
    for page_num in range(len(pdf_reader.pages)):
        pdf_text = pdf_reader.pages[page_num].extract_text()
        pdf_arr.append(pdf_text)
    return pdf_arr
# Streamlit app
st.title("PDF Explorer")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    # Extract text from uploaded PDF
    pdf_arr = extract_text_from_pdf(uploaded_file)
    
    st.write("PDF Uploaded Successfully.")
    
    # Text area for entering a question
    question = st.text_input("Ask a question about the PDF")
    pdf_text = find_most_relevant_context(pdf_arr,question)
    
    if st.button("Get Answer"):
        if question:
            # Get the answer from the backend
            answer = answer_question_from_pdf(pdf_text, question)
            st.write("Answer:", answer)
        else:
            st.write("Please enter a question.")
else:
    st.write("Please upload a PDF file.")
