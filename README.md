# RAG System

A modular Retrieval-Augmented Generation (RAG) system for document-based question answering.

## Features

- **Document Processing**: Extract and chunk text from PDF, DOCX, and TXT files
- **Semantic Search**: Embed and search documents based on meaning, not just keywords
- **Flexible Architecture**: Support for multiple embedding models and vector databases
- **REST API**: API for integrating with other applications
- **Web UI**: User-friendly Streamlit interface for document upload and querying

## Architecture

The system consists of the following components:

- **Embedding Model**: Converts text to vector embeddings
- **Vector Database**: Stores and searches document embeddings
- **Document Processor**: Extracts and chunks text from documents
- **RAG Engine**: Combines retrieval and generation for question answering
- **API**: Exposes functionality through a RESTful API
- **UI**: Provides a user interface for interacting with the system

## Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/rag-system.git
   cd rag-system
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables (optional):
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

## Usage

### API Server

Run the API server:

```bash
python app.py
```

The API will be available at http://localhost:8000

### Streamlit UI

Run the Streamlit UI:

```bash
streamlit run ui/app.py
```

The UI will be available at http://localhost:8501

## API Endpoints

- `POST /documents`: Add documents
- `POST /upload`: Upload and process document files
- `POST /query`: Query the RAG system
- `GET /search`: Search for documents
- `DELETE /documents`: Clear all documents
- `GET /health`: Check system health

## Configuration

The system can be configured through environment variables or the `config.py` file:

- `EMBEDDING_MODEL_NAME`: Name of the embedding model
- `VECTOR_DB_TYPE`: Type of vector database to use
- `CHUNK_SIZE`: Size of document chunks
- `CHUNK_OVERLAP`: Overlap between chunks
- `TOP_K`: Number of documents to retrieve
- `SEARCH_TYPE`: Type of search (semantic, keyword, hybrid)
- `LLM_MODEL_NAME`: Name of the language model for generation
- `LLM_API_KEY`: API key for the language model

## Extending

The modular architecture makes it easy to extend the system:

- Add new embedding models in `embedding/model.py`
- Add new vector databases in `storage/vector_db.py`
- Add support for new document types in `document/processor.py`
- Add new LLM integrations in `llm/model.py`

## License

[MIT License](LICENSE)
