"""
Centralized configuration for the RAG system.
"""

import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
# LLM settings
LLM_MODEL_NAME = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
LLM_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "512"))

# HuggingFace API settings
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
HUGGINGFACE_MODEL = os.getenv("HUGGINGFACE_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
# Embedding model settings
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "384"))
USE_GPU = os.getenv("USE_GPU", "True").lower() in ("true", "1", "t")

# Document processing settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "512"))

# Vector database settings
VECTOR_DB_TYPE = os.getenv("VECTOR_DB_TYPE", "faiss")  # Options: "faiss", "milvus", etc.
FAISS_INDEX_TYPE = os.getenv("FAISS_INDEX_TYPE", "Flat")  # Options: "Flat", "IVF", "HNSW"
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("DB_NAME", "rag_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "documents")

# Retrieval settings
TOP_K = int(os.getenv("TOP_K", "5"))
SEARCH_TYPE = os.getenv("SEARCH_TYPE", "hybrid")  # Options: "semantic", "keyword", "hybrid"
SEMANTIC_SEARCH_WEIGHT = float(os.getenv("SEMANTIC_SEARCH_WEIGHT", "0.7"))
KEYWORD_SEARCH_WEIGHT = float(os.getenv("KEYWORD_SEARCH_WEIGHT", "0.3"))

# LLM settings
LLM_MODEL_NAME = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
LLM_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "512"))

# Local LLM settings (optional)
LOCAL_LLM_MODEL_NAME = os.getenv("LOCAL_LLM_MODEL", "google/flan-t5-base")
USE_LOCAL_LLM = os.getenv("USE_LOCAL_LLM", "False").lower() in ("true", "1", "t")

# API settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# Logging settings
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Default prompt template
DEFAULT_PROMPT_TEMPLATE = """
Answer the following question based ONLY on the provided context. 
If you cannot answer the question based on the context, say "I don't have enough information to answer this question."

Context:
{context}

Question: {query}

Answer:
"""


def get_logging_config() -> Dict[str, Any]:
    """Get logging configuration dictionary."""
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": LOG_FORMAT
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": LOG_LEVEL,
                "formatter": "standard",
                "stream": "ext://sys.stdout"
            },
        },
        "loggers": {
            "": {
                "handlers": ["console"],
                "level": LOG_LEVEL,
                "propagate": True
            }
        }
    }


def get_model_config(model_name: Optional[str] = None) -> Dict[str, Any]:
    """Get model-specific configuration."""
    # Default to the configured model if none specified
    if model_name is None:
        model_name = EMBEDDING_MODEL_NAME
        
    # Common configurations for popular models
    config_map = {
        "sentence-transformers/all-MiniLM-L6-v2": {
            "dimension": 384,
            "max_length": 512,
            "normalize": True,
        },
        "sentence-transformers/all-mpnet-base-v2": {
            "dimension": 768,
            "max_length": 512,
            "normalize": True,
        },
        # Add more models as needed
    }
    
    # Return specific config if available, otherwise return default values
    return config_map.get(model_name, {
        "dimension": EMBEDDING_DIMENSION,
        "max_length": MAX_LENGTH,
        "normalize": True,
    })
