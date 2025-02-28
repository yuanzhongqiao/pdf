"""
FastAPI routes for the RAG system API.
"""

import logging
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Query, Depends, File, UploadFile
from pydantic import BaseModel, Field
import tempfile
import os
import time

# Configure logging
logger = logging.getLogger(__name__)


# Define Pydantic models for API requests and responses
class DocumentInput(BaseModel):
    """Input model for adding a document."""
    title: str
    text: str
    source: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentsInput(BaseModel):
    """Input model for adding multiple documents."""
    documents: List[DocumentInput]


class QueryInput(BaseModel):
    """Input model for querying the RAG system."""
    query: str
    top_k: int = 5
    search_type: str = "hybrid"
    filter_dict: Optional[Dict[str, Any]] = None
    max_tokens: int = 512


class SearchResult(BaseModel):
    """Model for search results."""
    id: str
    text: str
    metadata: Dict[str, Any]
    score: float


class RAGResponse(BaseModel):
    """Model for RAG system responses."""
    query: str
    response: str
    retrieved_documents: List[SearchResult]
    search_type: str


class HealthResponse(BaseModel):
    """Model for system health check responses."""
    status: str
    version: str
    document_count: int
    message: Optional[str] = None


# Class to define API routes
class RAGAPIRouter:
    """FastAPI router for the RAG system API."""
    
    def __init__(self, app: FastAPI, rag_engine):
        """
        Initialize the API router.
        
        Args:
            app: FastAPI application
            rag_engine: RAG engine instance
        """
        self.app = app
        self.rag_engine = rag_engine
        
        # Register routes
        self._register_routes()
    
    def _register_routes(self):
        """Register all API routes."""
        
        @self.app.post("/documents", response_model=Dict[str, Any], summary="Add documents to the system")
        async def add_documents(documents: DocumentsInput):
            """
            Add documents to the search index.
            
            - **documents**: List of documents with text and metadata
            
            Returns:
                Status and number of documents added
            """
            try:
                # Extract text and metadata
                texts = [doc.text for doc in documents.documents]
                metadata = [
                    {
                        **doc.metadata,
                        "title": doc.title,
                        "source": doc.source or "API upload"
                    }
                    for doc in documents.documents
                ]
                
                # Add to RAG engine
                doc_ids = self.rag_engine.add_documents(texts, metadata)
                
                return {
                    "status": "success",
                    "message": f"Added {len(doc_ids)} documents",
                    "document_ids": doc_ids
                }
            except Exception as e:
                logger.error(f"Error adding documents: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to add documents: {str(e)}")
        
        @self.app.post("/upload", response_model=Dict[str, Any], summary="Upload and process document files")
        async def upload_document(
            file: UploadFile = File(...),
            chunk_size: int = Query(1000, ge=100, le=5000),
            chunk_overlap: int = Query(200, ge=0, le=500)
        ):
            """
            Upload and process a document file.
            
            - **file**: Document file to upload (PDF, TXT, DOCX, etc.)
            - **chunk_size**: Size of text chunks
            - **chunk_overlap**: Overlap between chunks
            
            Returns:
                Status and number of chunks extracted
            """
            try:
                # Import here to avoid circular imports
                from document.processor import DocumentProcessor
                
                start_time = time.time()
                # Create temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
                    # Write uploaded file to temp file
                    content = await file.read()
                    temp_file.write(content)
                    temp_path = temp_file.name
                
                try:
                    # Process the file
                    processor = DocumentProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                    chunks, chunk_metadata = processor.process_file(
                        temp_path,
                        metadata={"filename": file.filename, "source": "API upload"}
                    )
                    
                    if not chunks:
                        return {
                            "status": "warning",
                            "message": "No text extracted from document"
                        }
                    
                    # Add chunks to RAG engine
                    doc_ids = self.rag_engine.add_documents(chunks, chunk_metadata)
                    
                    return {
                        "status": "success",
                        "message": f"Processed document into {len(chunks)} chunks",
                        "document_ids": doc_ids,
                        "processing_time_seconds": round(time.time() - start_time, 2)
                    }
                finally:
                    # Clean up temporary file
                    os.unlink(temp_path)
            except Exception as e:
                logger.error(f"Error processing document: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")
        
        @self.app.post("/query", response_model=RAGResponse, summary="Query the RAG system")
        async def query(query_input: QueryInput):
            """
            Generate a response for the query using the RAG system.
            
            - **query**: Query text
            - **top_k**: Number of documents to retrieve
            - **search_type**: Type of search ('semantic', 'keyword', 'hybrid')
            - **filter_dict**: Optional metadata filters
            - **max_tokens**: Maximum tokens in response
            
            Returns:
                RAG response with answer and retrieved documents
            """
            try:
                # Query RAG engine
                result = self.rag_engine.generate_response(
                    query=query_input.query,
                    top_k=query_input.top_k,
                    search_type=query_input.search_type,
                    filter_dict=query_input.filter_dict,
                    max_tokens=query_input.max_tokens
                )
                
                # Convert to response model
                return RAGResponse(
                    query=result["query"],
                    response=result["response"],
                    retrieved_documents=[
                        SearchResult(
                            id=doc["id"],
                            text=doc["text"],
                            metadata=doc["metadata"],
                            score=doc["score"]
                        )
                        for doc in result["retrieved_documents"]
                    ],
                    search_type=result["search_type"]
                )
            except Exception as e:
                logger.error(f"Error querying RAG system: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to query system: {str(e)}")
        
        @self.app.get("/search", summary="Search for documents without generating a response")
        async def search(
            query: str,
            top_k: int = Query(5, ge=1, le=20),
            search_type: str = Query("hybrid", regex="^(semantic|keyword|hybrid)$")
        ):
            """
            Search for documents without generating a response.
            
            - **query**: Search query
            - **top_k**: Number of results to return
            - **search_type**: Type of search
            
            Returns:
                Search results
            """
            try:
                # Search for documents
                results = self.rag_engine.search(
                    query=query,
                    top_k=top_k,
                    search_type=search_type
                )
                
                return {
                    "query": query,
                    "results": results,
                    "search_type": search_type,
                    "count": len(results)
                }
            except Exception as e:
                logger.error(f"Error searching documents: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to search documents: {str(e)}")
        
        @self.app.delete("/documents", summary="Clear all documents from the system")
        async def clear_documents():
            """
            Clear all documents from the system.
            
            Returns:
                Status message
            """
            try:
                # Clear documents
                self.rag_engine.clear_documents()
                
                return {
                    "status": "success",
                    "message": "All documents cleared from the system"
                }
            except Exception as e:
                logger.error(f"Error clearing documents: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to clear documents: {str(e)}")
        
        @self.app.get("/health", response_model=HealthResponse, summary="Check system health")
        async def health_check():
            """
            Check if the system is healthy.
            
            Returns:
                System health status
            """
            try:
                # Get document count
                doc_count = self.rag_engine.count_documents()
                
                return HealthResponse(
                    status="healthy",
                    version="1.0.0",
                    document_count=doc_count,
                    message="System is operational"
                )
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                return HealthResponse(
                    status="unhealthy",
                    version="1.0.0",
                    document_count=0,
                    message=f"System error: {str(e)}"
                )