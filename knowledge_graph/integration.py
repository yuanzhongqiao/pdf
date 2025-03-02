"""
Integration of knowledge graph with the RAG system.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Set, Union
import re

from .model import KnowledgeGraph, Entity, Relation
from .extractor import KnowledgeGraphExtractor

# Configure logging
logger = logging.getLogger(__name__)


class KnowledgeGraphEnhancedRetrieval:
    """
    Enhances document retrieval using knowledge graph.
    """
    
    def __init__(
        self, 
        knowledge_graph: KnowledgeGraph,
        rag_engine = None,
        use_for_query_expansion: bool = True,
        use_for_results_ranking: bool = True
    ):
        """
        Initialize knowledge graph enhanced retrieval.
        
        Args:
            knowledge_graph: Knowledge graph to use
            rag_engine: RAG engine to enhance
            use_for_query_expansion: Whether to use knowledge graph for query expansion
            use_for_results_ranking: Whether to use knowledge graph for results ranking
        """
        self.knowledge_graph = knowledge_graph
        self.rag_engine = rag_engine
        self.use_for_query_expansion = use_for_query_expansion
        self.use_for_results_ranking = use_for_results_ranking
    
    def expand_query(self, query: str) -> str:
        """
        Expand a query using knowledge graph relationships.
        
        Args:
            query: Original query
            
        Returns:
            Expanded query
        """
        if not self.use_for_query_expansion:
            return query
            
        logger.info(f"Expanding query: '{query}'")
        
        # Extract entities from query
        query_entities = self._extract_entities_from_query(query)
        logger.info(f"Found {len(query_entities)} entities in query")
        
        if not query_entities:
            return query
        
        # Expand query with related entities
        expansion_terms = []
        
        for entity_name, entity_type in query_entities:
            # Find matching entities in knowledge graph
            matches = self.knowledge_graph.get_entity_by_name(entity_name)
            
            if not matches:
                continue
                
            for entity in matches:
                # Get neighbors
                neighbors = self.knowledge_graph.get_neighbors(entity.id)
                
                # Add neighbor names as expansion terms
                for neighbor, relation in neighbors:
                    expansion_terms.append(neighbor.name)
        
        # Deduplicate expansion terms
        expansion_terms = list(set(expansion_terms))
        logger.info(f"Found {len(expansion_terms)} expansion terms")
        
        # Limit number of expansion terms
        expansion_terms = expansion_terms[:5]
        
        # Create expanded query
        if expansion_terms:
            expanded_query = f"{query} {' '.join(expansion_terms)}"
            logger.info(f"Expanded query: '{expanded_query}'")
            return expanded_query
        else:
            return query
    
    def rerank_results(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rerank results using knowledge graph relevance.
        
        Args:
            query: Original query
            results: List of result dictionaries
            
        Returns:
            Reranked results
        """
        if not self.use_for_results_ranking or not results:
            return results
            
        logger.info(f"Reranking {len(results)} results using knowledge graph")
        
        # Extract entities from query
        query_entities = self._extract_entities_from_query(query)
        
        if not query_entities:
            return results
            
        # Extract entities from results
        result_entities = {}
        for i, result in enumerate(results):
            text = result.get("text", "")
            extracted = self._extract_entities_from_text(text)
            result_entities[i] = extracted
        
        # Calculate graph-based relevance scores
        graph_scores = {}
        
        for i, entities in result_entities.items():
            # Find connections between query entities and result entities
            connection_score = self._calculate_connection_score(query_entities, entities)
            
            # Store graph score
            graph_scores[i] = connection_score
        
        # Combine original scores with graph scores
        for i, result in enumerate(results):
            original_score = result.get("score", 0.0)
            graph_score = graph_scores.get(i, 0.0)
            
            # Weighted combination
            combined_score = 0.7 * original_score + 0.3 * graph_score
            
            # Update score
            result["score"] = combined_score
            result["graph_score"] = graph_score
        
        # Rerank based on combined scores
        reranked_results = sorted(results, key=lambda x: x.get("score", 0.0), reverse=True)
        
        return reranked_results
    
    def enhance_search(self, query: str, top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """
        Perform enhanced search using knowledge graph.
        
        Args:
            query: Query string
            top_k: Number of results to return
            **kwargs: Additional search parameters
            
        Returns:
            List of search results
        """
        if not self.rag_engine:
            logger.warning("No RAG engine provided, cannot perform enhanced search")
            return []
        
        # Expand query if enabled
        expanded_query = self.expand_query(query) if self.use_for_query_expansion else query
        
        # Perform search with expanded query
        results = self.rag_engine.search(expanded_query, top_k=top_k, **kwargs)
        
        # Rerank results if enabled
        if self.use_for_results_ranking:
            results = self.rerank_results(query, results)
        
        return results
    
    def enhance_response_generation(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Generate an enhanced response using knowledge graph.
        
        Args:
            query: Query string
            **kwargs: Additional parameters for response generation
            
        Returns:
            Response dictionary
        """
        if not self.rag_engine:
            logger.warning("No RAG engine provided, cannot generate enhanced response")
            return {"query": query, "response": "No RAG engine available for response generation"}
        
        # Expand query if enabled
        expanded_query = self.expand_query(query) if self.use_for_query_expansion else query
        
        # Generate response
        response = self.rag_engine.generate_response(expanded_query, **kwargs)
        
        # Add knowledge graph context if available
        query_entities = self._extract_entities_from_query(query)
        
        if query_entities and "additional_context" not in response:
            response["additional_context"] = self._get_knowledge_graph_context(query_entities)
        
        return response
    
    def _extract_entities_from_query(self, query: str) -> List[Tuple[str, str]]:
        """
        Extract entities from a query.
        
        Args:
            query: Query string
            
        Returns:
            List of (entity_name, entity_type) tuples
        """
        # Simple noun phrase extraction for now
        noun_phrases = []
        
        # Use regex to find potential noun phrases
        # This is a simplified approach; a proper NLP parser would be better
        noun_pattern = r'[A-Z][a-z]+(?:\s+[A-Za-z]+)*|[a-z]+(?:\s+[a-z]+)+'
        matches = re.finditer(noun_pattern, query)
        
        for match in matches:
            phrase = match.group()
            if len(phrase) > 2 and phrase.lower() not in ["the", "a", "an", "this", "that", "these", "those"]:
                noun_phrases.append((phrase, "Unknown"))
        
        return noun_phrases
    
    def _extract_entities_from_text(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract entities from text.
        
        Args:
            text: Text to extract from
            
        Returns:
            List of (entity_name, entity_type) tuples
        """
        # Simple extraction using the same approach as for queries
        return self._extract_entities_from_query(text)
    
    def _calculate_connection_score(self, query_entities: List[Tuple[str, str]], result_entities: List[Tuple[str, str]]) -> float:
        """
        Calculate connection score between query entities and result entities.
        
        Args:
            query_entities: List of (entity_name, entity_type) tuples from query
            result_entities: List of (entity_name, entity_type) tuples from result
            
        Returns:
            Connection score
        """
        if not query_entities or not result_entities:
            return 0.0
        
        # Find knowledge graph entities for query entities
        query_kg_entities = []
        for entity_name, _ in query_entities:
            matches = self.knowledge_graph.get_entity_by_name(entity_name)
            query_kg_entities.extend(matches)
        
        # Find knowledge graph entities for result entities
        result_kg_entities = []
        for entity_name, _ in result_entities:
            matches = self.knowledge_graph.get_entity_by_name(entity_name)
            result_kg_entities.extend(matches)
        
        # If no entities found in knowledge graph, return 0
        if not query_kg_entities or not result_kg_entities:
            return 0.0
        
        # Calculate connection score based on paths between entities
        connection_count = 0
        total_paths = 0
        
        for query_entity in query_kg_entities:
            for result_entity in result_kg_entities:
                # Find paths between entities
                paths = self.knowledge_graph.find_path(query_entity.id, result_entity.id, max_hops=2)
                
                if paths:
                    connection_count += 1
                    total_paths += len(paths)
        
        # Normalize score
        max_possible_connections = len(query_kg_entities) * len(result_kg_entities)
        if max_possible_connections > 0:
            connection_score = connection_count / max_possible_connections
        else:
            connection_score = 0.0
        
        # Boost score based on number of paths
        if connection_count > 0:
            path_factor = min(1.0, total_paths / (connection_count * 3))  # Cap at 1.0
            connection_score = connection_score * (1.0 + path_factor)
        
        return connection_score
    
    def _get_knowledge_graph_context(self, query_entities: List[Tuple[str, str]]) -> Dict[str, Any]:
        """
        Get relevant knowledge graph context for query entities.
        
        Args:
            query_entities: List of (entity_name, entity_type) tuples from query
            
        Returns:
            Dictionary of knowledge graph context
        """
        context = {
            "entities": [],
            "relations": []
        }
        
        # Find knowledge graph entities for query entities
        for entity_name, _ in query_entities:
            matches = self.knowledge_graph.get_entity_by_name(entity_name)
            
            for entity in matches:
                # Add entity info
                entity_info = {
                    "name": entity.name,
                    "type": entity.type,
                    "id": entity.id
                }
                
                # Get connections
                connections = self.knowledge_graph.get_entity_connections(entity.id)
                
                # Format connections
                formatted_connections = {}
                for relation_type, neighbors in connections.items():
                    formatted_connections[relation_type] = [
                        {"name": neighbor.name, "type": neighbor.type, "id": neighbor.id}
                        for neighbor, _ in neighbors
                    ]
                
                entity_info["connections"] = formatted_connections
                context["entities"].append(entity_info)
        
        return context


class RAGEnhancer:
    """
    Enhances a RAG system with knowledge graph capabilities.
    """
    
    def __init__(self, rag_engine):
        """
        Initialize the RAG enhancer.
        
        Args:
            rag_engine: RAG engine to enhance
        """
        self.rag_engine = rag_engine
        self.knowledge_graph = KnowledgeGraph()
        self.extractor = KnowledgeGraphExtractor()
        self.retrieval_enhancer = None
    
    def build_knowledge_graph(self, reset: bool = True) -> None:
        """
        Build knowledge graph from documents in the RAG engine.
        
        Args:
            reset: Whether to reset existing knowledge graph
        """
        if reset:
            self.knowledge_graph = KnowledgeGraph()
            self.extractor.reset()
        
        # Get document chunks from the RAG engine's vector database
        documents = []
        metadata = []
        
        # Access vector database through RAG engine
        if hasattr(self.rag_engine, "vector_db") and hasattr(self.rag_engine.vector_db, "documents"):
            # Get documents from vector database
            for doc_id, doc in self.rag_engine.vector_db.documents.items():
                documents.append(doc.text)
                metadata.append(doc.metadata)
        
        if not documents:
            logger.warning("No documents found in RAG engine")
            return
        
        logger.info(f"Building knowledge graph from {len(documents)} documents")
        
        # Process document chunks
        self.extractor.process_document_chunks(documents, metadata)
        
        # Update knowledge graph
        self.knowledge_graph = self.extractor.get_knowledge_graph()
        
        # Create retrieval enhancer
        self.retrieval_enhancer = KnowledgeGraphEnhancedRetrieval(
            knowledge_graph=self.knowledge_graph,
            rag_engine=self.rag_engine
        )
        
        # Log statistics
        stats = self.knowledge_graph.statistics()
        logger.info(f"Knowledge graph built with {stats['num_entities']} entities and {stats['num_relations']} relations")
    
    def add_documents(self, texts: List[str], metadata: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """
        Add documents to both RAG engine and knowledge graph.
        
        Args:
            texts: List of text chunks
            metadata: Optional list of metadata dictionaries
            
        Returns:
            List of document IDs
        """
        # Add to RAG engine
        doc_ids = self.rag_engine.add_documents(texts, metadata)
        
        # Process for knowledge graph
        self.extractor.process_document_chunks(texts, metadata)
        
        # Update knowledge graph
        self.knowledge_graph = self.extractor.get_knowledge_graph()
        
        # Create retrieval enhancer if needed
        if self.retrieval_enhancer is None:
            self.retrieval_enhancer = KnowledgeGraphEnhancedRetrieval(
                knowledge_graph=self.knowledge_graph,
                rag_engine=self.rag_engine
            )
        
        return doc_ids
    
    def search(self, query: str, top_k: Optional[int] = None, search_type: Optional[str] = None, **kwargs) -> List[Dict[str, Any]]:
        """
        Search using knowledge graph enhanced retrieval.
        
        Args:
            query: Query string
            top_k: Number of results to return
            search_type: Type of search
            **kwargs: Additional search parameters
            
        Returns:
            List of search results
        """
        # If retrieval enhancer not initialized, create it
        if self.retrieval_enhancer is None:
            self.retrieval_enhancer = KnowledgeGraphEnhancedRetrieval(
                knowledge_graph=self.knowledge_graph,
                rag_engine=self.rag_engine
            )
        
        # Perform enhanced search
        return self.retrieval_enhancer.enhance_search(query, top_k=top_k, search_type=search_type, **kwargs)
    
    def generate_response(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Generate a response using knowledge graph enhanced retrieval.
        
        Args:
            query: Query string
            **kwargs: Additional parameters for response generation
            
        Returns:
            Response dictionary
        """
        # If retrieval enhancer not initialized, create it
        if self.retrieval_enhancer is None:
            self.retrieval_enhancer = KnowledgeGraphEnhancedRetrieval(
                knowledge_graph=self.knowledge_graph,
                rag_engine=self.rag_engine
            )
        
        # Generate enhanced response
        return self.retrieval_enhancer.enhance_response_generation(query, **kwargs)
    
    def get_knowledge_graph(self) -> KnowledgeGraph:
        """
        Get the knowledge graph.
        
        Returns:
            KnowledgeGraph instance
        """
        return self.knowledge_graph
    
    def visualize_knowledge_graph(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a visualization of the knowledge graph.
        
        Args:
            output_path: Optional path to save visualization
            
        Returns:
            Dictionary with graph data for visualization
        """
        # Generate graph data
        graph_data = {
            "nodes": [],
            "links": []
        }
        
        # Add nodes
        node_colors = {
            "Person": "#4285F4",  # Blue
            "Organization": "#EA4335",  # Red
            "Location": "#FBBC05",  # Yellow
            "Concept": "#34A853",  # Green
            "Product": "#8F00FF",  # Purple
            "Event": "#FF6D01",  # Orange
            "WorkOfArt": "#00BCD4",  # Cyan
            "Other": "#9E9E9E"  # Gray
        }
        
        for entity_id, entity in self.knowledge_graph.entities.items():
            color = node_colors.get(entity.type, node_colors["Other"])
            
            node = {
                "id": entity_id,
                "name": entity.name,
                "type": entity.type,
                "color": color,
                "size": 5
            }
            
            graph_data["nodes"].append(node)
        
        # Add links
        for relation_id, relation in self.knowledge_graph.relations.items():
            link = {
                "source": relation.source,
                "target": relation.target,
                "type": relation.type,
                "value": relation.weight
            }
            
            graph_data["links"].append(link)
        
        # Save visualization if output path provided
        if output_path:
            import json
            with open(output_path, 'w') as f:
                json.dump(graph_data, f, indent=2)
        
        return graph_data
    
    def save_knowledge_graph(self, file_path: str) -> None:
        """
        Save the knowledge graph to a file.
        
        Args:
            file_path: Path to save to
        """
        self.knowledge_graph.save(file_path)
    
    def load_knowledge_graph(self, file_path: str) -> None:
        """
        Load a knowledge graph from a file.
        
        Args:
            file_path: Path to load from
        """
        self.knowledge_graph = KnowledgeGraph.load(file_path)
        
        # Create retrieval enhancer
        self.retrieval_enhancer = KnowledgeGraphEnhancedRetrieval(
            knowledge_graph=self.knowledge_graph,
            rag_engine=self.rag_engine
        )