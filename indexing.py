from typing import List, Dict, Any, Optional, Union
import numpy as np
from pymongo import MongoClient, ASCENDING
from pymongo.database import Database
from pymongo.collection import Collection
import uuid
import os

# Import memory database for Hugging Face Spaces demo
from memory_db import MemoryClient


class DocumentIndexer:
    """
    Handles indexing and retrieval of document chunks with vector search.
    """

    def __init__(
        self,
        db_uri: str = "mongodb://localhost:27017/",
        db_name: str = "rag_db",
        collection_name: str = "documents"
    ):
        """
        Initialize the document indexer.

        Args:
            db_uri: MongoDB connection URI (or 'memory://' for in-memory database)
            db_name: Database name
            collection_name: Collection name for document chunks
        """
        # Use in-memory database if specified
        if db_uri.startswith("memory://"):
            self.client = MemoryClient(db_uri)
        else:
            self.client = MongoClient(db_uri)

        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

        # Create indexes if using MongoDB
        if not db_uri.startswith("memory://"):
            self.collection.create_index([("embedding", ASCENDING)], sparse=True)
            self.collection.create_index([("metadata.source", ASCENDING)], sparse=True)

    def index_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Index document chunks with their embeddings.

        Args:
            chunks: List of document chunks with text, embeddings, and metadata

        Returns:
            Dict with count of indexed chunks
        """
        # Add unique IDs to chunks if not present
        for chunk in chunks:
            if "id" not in chunk:
                chunk["id"] = str(uuid.uuid4())

        # Insert chunks into database
        result = self.collection.insert_many(chunks)

        return {
            "indexed_count": len(result.inserted_ids) if hasattr(result, "inserted_ids") else len(chunks)
        }

    def vector_search(
        self,
        query_embedding: List[float],
        limit: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search.

        Args:
            query_embedding: Vector embedding of the query
            limit: Maximum number of results to return
            metadata_filter: Optional filter for metadata fields

        Returns:
            List of matching documents with similarity scores
        """
        filter_dict = {}
        if metadata_filter:
            for key, value in metadata_filter.items():
                filter_dict[f"metadata.{key}"] = value

        # Check if we're using memory DB or MongoDB
        if isinstance(self.client, MemoryClient):
            # Use the memory DB vector search
            results = self.collection.vector_search(
                vector=query_embedding,
                limit=limit,
                filter_dict=filter_dict
            )
        else:
            # Use MongoDB's vector search
            # For MongoDB, we need to implement the vector search differently
            # This is a simplified approach - in production, use MongoDB Atlas Vector Search
            pipeline = []

            # Add metadata filters if provided
            if filter_dict:
                pipeline.append({"$match": filter_dict})

            # Get all documents with embeddings
            pipeline.append({"$match": {"embedding": {"$exists": True}}})

            # Convert results to our format with similarity scores
            results = list(self.collection.aggregate(pipeline))

            # Calculate cosine similarity manually
            query_embedding_np = np.array(query_embedding)
            for result in results:
                doc_embedding = np.array(result["embedding"])
                similarity = np.dot(query_embedding_np, doc_embedding) / (
                    np.linalg.norm(query_embedding_np) * np.linalg.norm(doc_embedding)
                )
                result["similarity"] = float(similarity)

            # Sort by similarity and limit results
            results.sort(key=lambda x: x["similarity"], reverse=True)
            results = results[:limit]
        return results

    def keyword_search(
        self,
        query: str,
        limit: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform keyword-based search.

        Args:
            query: Search query string
            limit: Maximum number of results to return
            metadata_filter: Optional filter for metadata fields

        Returns:
            List of matching documents
        """
        # For this simplified implementation, we'll just do a text match
        # In a real application, you would use a proper text search index

        filter_dict = {"text": {"$regex": query, "$options": "i"}}
        if metadata_filter:
            for key, value in metadata_filter.items():
                filter_dict[f"metadata.{key}"] = value

        results = list(self.collection.find(filter_dict).limit(limit))

        # Add a simple match score
        for result in results:
            # Simple scoring - just count occurrences of query terms
            query_terms = query.lower().split()
            doc_text = result["text"].lower()
            score = sum(doc_text.count(term) for term in query_terms)
            result["score"] = score

        # Sort by score
        results.sort(key=lambda x: x.get("score", 0), reverse=True)

        return results

    def delete_by_source(self, source: str) -> Dict[str, int]:
        """
        Delete all chunks from a specific source.

        Args:
            source: Source identifier (e.g., file path)

        Returns:
            Dict with count of deleted chunks
        """
        result = self.collection.delete_many({"metadata.source": source})
        return {"deleted_count": result.deleted_count}

    def get_all_sources(self) -> List[str]:
        """
        Get list of all document sources in the index.

        Returns:
            List of source identifiers
        """
        # Use distinct to get unique source values
        sources = list(self.collection.distinct("metadata.source"))
        return sources

    def close(self):
        """Close the database connection."""
        self.client.close()
