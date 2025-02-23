# config.py
CHUNK_SIZE = 500
OVERLAP = 50
TOP_K = 5
BATCH_SIZE = 32
EMBEDDING_MODEL = "BAAI/bge-large-en"
RERANKER_MODEL = "BAAI/bge-reranker-large"
API_URL_MISTRAL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"