from typing import List, Dict, Any, Optional, Union
import uuid
import numpy as np


class MemoryCollection:
    """Mimics a MongoDB collection but stores data in memory."""

    def __init__(self):
        self.documents = []
        self.index = {}  # Simple index for vector search

    def insert_one(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Insert a document into the collection."""
        if "id" not in document:
            document["id"] = str(uuid.uuid4())
        self.documents.append(document)

        # Index vector if present
        if "embedding" in document:
            self.index[document["id"]] = document["embedding"]

        return {"id": document["id"]}

    def insert_many(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Insert multiple documents into the collection."""
        for doc in documents:
            self.insert_one(doc)
        return {"inserted_count": len(documents)}

    def find(self, query: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Find documents matching the query."""
        if query is None:
            return self.documents.copy()

        results = []
        for doc in self.documents:
            match = True
            for key, value in query.items():
                if key not in doc or doc[key] != value:
                    match = False
                    break
            if match:
                results.append(doc.copy())

        return results

    def find_one(self, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find a single document matching the query."""
        results = self.find(query)
        return results[0] if results else None

    def create_index(self, keys: List[str], **kwargs) -> None:
        """Mock creating an index (no-op for in-memory)."""
        pass

    def vector_search(self, vector: List[float], limit: int = 5,
                      filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Perform vector search on embeddings."""
        if not self.index:
            return []

        vector_np = np.array(vector)
        results = []

        for doc in self.documents:
            if "embedding" not in doc:
                continue

            # Apply filter if present
            if filter_dict:
                match = True
                for key, value in filter_dict.items():
                    if key not in doc or doc[key] != value:
                        match = False
                        break
                if not match:
                    continue

            # Calculate cosine similarity
            doc_vector = np.array(doc["embedding"])
            similarity = np.dot(vector_np, doc_vector) / (
                np.linalg.norm(vector_np) * np.linalg.norm(doc_vector)
            )

            results.append({
                **doc.copy(),
                "similarity": float(similarity)
            })

        # Sort by similarity and return top results
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:limit]

    def delete_many(self, query: Dict[str, Any]) -> Dict[str, int]:
        """Delete documents matching the query."""
        initial_count = len(self.documents)

        self.documents = [
            doc for doc in self.documents
            if not all(doc.get(k) == v for k, v in query.items())
        ]

        # Update index
        self.index = {
            doc_id: embedding for doc_id, embedding in self.index.items()
            if any(doc["id"] == doc_id for doc in self.documents)
        }

        deleted = initial_count - len(self.documents)
        return {"deleted_count": deleted}


class MemoryDatabase:
    """Mimics a MongoDB database but stores data in memory."""

    def __init__(self, name: str):
        self.name = name
        self.collections = {}

    def __getitem__(self, collection_name: str) -> MemoryCollection:
        if collection_name not in self.collections:
            self.collections[collection_name] = MemoryCollection()
        return self.collections[collection_name]

    def list_collection_names(self) -> List[str]:
        return list(self.collections.keys())


class MemoryClient:
    """Mimics a MongoDB client but stores data in memory."""

    def __init__(self, uri: str = "memory://"):
        self.uri = uri
        self.databases = {}

    def __getitem__(self, db_name: str) -> MemoryDatabase:
        if db_name not in self.databases:
            self.databases[db_name] = MemoryDatabase(db_name)
        return self.databases[db_name]

    def close(self) -> None:
        pass