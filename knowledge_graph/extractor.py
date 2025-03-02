"""
Entity and relation extraction for knowledge graph construction.
"""

import re
import logging
import spacy
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from collections import defaultdict

from .model import Entity, Relation, KnowledgeGraph

# Configure logging
logger = logging.getLogger(__name__)


class KnowledgeGraphExtractor:
    """
    Extracts entities and relations from text to build a knowledge graph.
    """
    
    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        load_ner: bool = True,
        load_custom_rules: bool = True
    ):
        """
        Initialize the extractor.
        
        Args:
            spacy_model: spaCy model to use
            load_ner: Whether to use NER from spaCy
            load_custom_rules: Whether to load custom extraction rules
        """
        self.load_spacy(spacy_model)
        self.use_ner = load_ner
        
        # Initialize knowledge graph
        self.knowledge_graph = KnowledgeGraph()
        
        # Custom extraction rules
        self.custom_rules = []
        if load_custom_rules:
            self.load_custom_rules()
    
    def load_spacy(self, model_name: str) -> None:
        """
        Load spaCy model.
        
        Args:
            model_name: Name of spaCy model to load
        """
        try:
            self.nlp = spacy.load(model_name)
            logger.info(f"Loaded spaCy model: {model_name}")
        except OSError:
            logger.warning(f"spaCy model {model_name} not found. Attempting to download...")
            try:
                # Try to download the model
                import subprocess
                subprocess.run(["python", "-m", "spacy", "download", model_name], check=True)
                self.nlp = spacy.load(model_name)
                logger.info(f"Downloaded and loaded spaCy model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to download spaCy model: {e}")
                # Use simple tokenization as fallback
                logger.warning("Using simple tokenization as fallback")
                self.nlp = spacy.blank("en")
                self.use_ner = False
    
    def load_custom_rules(self) -> None:
        """
        Load custom extraction rules.
        
        Custom rules are patterns for specific relations that can be extracted.
        """
        # Basic relation patterns
        self.custom_rules = [
            {
                "name": "is_a_relation",
                "patterns": [
                    r"(?P<subject>[A-Z][a-z]+(?:\s+[A-Za-z]+)*) is (?:a|an) (?P<object>[a-z]+(?:\s+[a-z]+)*)",
                    r"(?P<subject>[A-Z][a-z]+(?:\s+[A-Za-z]+)*) are (?P<object>[a-z]+(?:\s+[a-z]+)*)"
                ],
                "relation": "is_a"
            },
            {
                "name": "part_of_relation",
                "patterns": [
                    r"(?P<subject>[A-Z][a-z]+(?:\s+[A-Za-z]+)*) is part of (?P<object>[A-Za-z]+(?:\s+[A-Za-z]+)*)",
                    r"(?P<subject>[A-Z][a-z]+(?:\s+[A-Za-z]+)*) belongs to (?P<object>[A-Za-z]+(?:\s+[A-Za-z]+)*)"
                ],
                "relation": "part_of"
            },
            {
                "name": "has_property",
                "patterns": [
                    r"(?P<subject>[A-Z][a-z]+(?:\s+[A-Za-z]+)*) has (?P<object>[a-z]+(?:\s+[a-z]+)*)",
                    r"(?P<subject>[A-Z][a-z]+(?:\s+[A-Za-z]+)*) contains (?P<object>[a-z]+(?:\s+[a-z]+)*)"
                ],
                "relation": "has_property"
            },
            {
                "name": "works_with",
                "patterns": [
                    r"(?P<subject>[A-Z][a-z]+(?:\s+[A-Za-z]+)*) works with (?P<object>[A-Za-z]+(?:\s+[A-Za-z]+)*)",
                    r"(?P<subject>[A-Z][a-z]+(?:\s+[A-Za-z]+)*) uses (?P<object>[A-Za-z]+(?:\s+[A-Za-z]+)*)"
                ],
                "relation": "works_with"
            }
        ]
        
        # Compile patterns
        for rule in self.custom_rules:
            rule["compiled_patterns"] = [re.compile(pattern) for pattern in rule["patterns"]]
    
    def extract_entities_from_text(self, text: str) -> List[Entity]:
        """
        Extract entities from text using spaCy NER and rules.
        
        Args:
            text: Text to extract entities from
            
        Returns:
            List of extracted entities
        """
        entities = []
        
        if not text or not text.strip():
            return entities
        
        # Process text with spaCy
        doc = self.nlp(text)
        
        # Extract named entities
        if self.use_ner:
            for ent in doc.ents:
                # Map spaCy entity types to our simplified types
                entity_type = self._map_entity_type(ent.label_)
                
                # Create entity
                entity = Entity(
                    name=ent.text,
                    type=entity_type,
                    metadata={
                        "source_text": text,
                        "spacy_label": ent.label_,
                        "start_char": ent.start_char,
                        "end_char": ent.end_char
                    }
                )
                
                entities.append(entity)
        
        # Extract noun phrases as concept entities
        for chunk in doc.noun_chunks:
            # Clean the noun phrase
            name = chunk.text.strip()
            
            # Skip short or non-informative phrases
            if len(name) < 3 or name.lower() in ["the", "a", "an", "this", "that", "these", "those", "i", "he", "she", "it", "they", "we", "you"]:
                continue
            
            # Create entity
            entity = Entity(
                name=name,
                type="Concept",
                metadata={
                    "source_text": text,
                    "is_noun_phrase": True,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char
                }
            )
            
            entities.append(entity)
        
        # Deduplicate entities
        unique_entities = []
        seen_names = set()
        
        for entity in entities:
            key = (entity.name.lower(), entity.type)
            if key not in seen_names:
                seen_names.add(key)
                unique_entities.append(entity)
        
        return unique_entities
    
    def extract_relations_from_text(self, text: str, entities: List[Entity]) -> List[Tuple[Entity, str, Entity]]:
        """
        Extract relations between entities in text.
        
        Args:
            text: Source text
            entities: Entities extracted from the text
            
        Returns:
            List of (source_entity, relation_type, target_entity) tuples
        """
        relations = []
        
        if not text or not text.strip() or len(entities) < 2:
            return relations
        
        # Create a mapping of entity names to entities
        entity_map = {}
        for entity in entities:
            entity_map[entity.name] = entity
            # Also add lowercase version for matching
            entity_map[entity.name.lower()] = entity
        
        # Apply custom rules for relation extraction
        for rule in self.custom_rules:
            for pattern in rule["compiled_patterns"]:
                for match in pattern.finditer(text):
                    if "subject" in match.groupdict() and "object" in match.groupdict():
                        subj_text = match.group("subject")
                        obj_text = match.group("object")
                        
                        # Find matching entities
                        subj_entity = None
                        obj_entity = None
                        
                        # Direct match
                        if subj_text in entity_map:
                            subj_entity = entity_map[subj_text]
                        
                        if obj_text in entity_map:
                            obj_entity = entity_map[obj_text]
                        
                        # Fuzzy match if direct match fails
                        if subj_entity is None:
                            for entity in entities:
                                if subj_text.lower() in entity.name.lower() or entity.name.lower() in subj_text.lower():
                                    subj_entity = entity
                                    break
                        
                        if obj_entity is None:
                            for entity in entities:
                                if obj_text.lower() in entity.name.lower() or entity.name.lower() in obj_text.lower():
                                    obj_entity = entity
                                    break
                        
                        # If both entities found, add relation
                        if subj_entity and obj_entity:
                            relations.append((subj_entity, rule["relation"], obj_entity))
        
        # Extract subject-verb-object relations
        doc = self.nlp(text)
        
        for token in doc:
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                # Found a main verb
                verb = token.lemma_
                
                # Find subject
                subjects = []
                for child in token.children:
                    if "subj" in child.dep_:
                        # Include the full noun phrase if possible
                        if child.i < len(doc) - 1:
                            full_subject = doc[child.left_edge.i:child.right_edge.i + 1].text
                        else:
                            full_subject = child.text
                        
                        subjects.append(full_subject)
                
                # Find object
                objects = []
                for child in token.children:
                    if "obj" in child.dep_:
                        # Include the full noun phrase if possible
                        if child.i < len(doc) - 1:
                            full_object = doc[child.left_edge.i:child.right_edge.i + 1].text
                        else:
                            full_object = child.text
                        
                        objects.append(full_object)
                
                # Create relations between subjects and objects
                for subj in subjects:
                    for obj in objects:
                        # Find matching entities
                        subj_entity = None
                        obj_entity = None
                        
                        # Try to match with entities
                        for entity in entities:
                            if entity.name == subj or entity.name in subj or subj in entity.name:
                                subj_entity = entity
                                break
                        
                        for entity in entities:
                            if entity.name == obj or entity.name in obj or obj in entity.name:
                                obj_entity = entity
                                break
                        
                        # If both entities found, add relation
                        if subj_entity and obj_entity:
                            relations.append((subj_entity, verb, obj_entity))
        
        return relations
    
    def process_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Process text and add extracted entities and relations to the knowledge graph.
        
        Args:
            text: Text to process
            metadata: Optional metadata about the source
        """
        logger.info(f"Processing text of length {len(text)}")
        
        # Extract entities
        entities = self.extract_entities_from_text(text)
        logger.info(f"Extracted {len(entities)} entities")
        
        # Add entities to graph
        for entity in entities:
            # Add metadata if provided
            if metadata:
                entity.metadata.update(metadata)
            
            self.knowledge_graph.add_entity(entity)
        
        # Extract relations
        relation_tuples = self.extract_relations_from_text(text, entities)
        logger.info(f"Extracted {len(relation_tuples)} relations")
        
        # Add relations to graph
        for source_entity, relation_type, target_entity in relation_tuples:
            relation = Relation(
                source=source_entity.id,
                target=target_entity.id,
                type=relation_type,
                metadata=metadata or {}
            )
            
            self.knowledge_graph.add_relation(relation)
        logger.info("Processing complete")
    
    def process_document_chunks(self, chunks: List[str], metadata_list: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Process document chunks and build a knowledge graph.
        
        Args:
            chunks: List of text chunks
            metadata_list: Optional list of metadata dictionaries
        """
        if metadata_list and len(metadata_list) != len(chunks):
            logger.warning(f"Length mismatch: {len(chunks)} chunks but {len(metadata_list)} metadata entries")
            metadata_list = None
        
        for i, chunk in enumerate(chunks):
            # Get metadata for this chunk
            chunk_metadata = None
            if metadata_list:
                chunk_metadata = metadata_list[i]
                
                # Add chunk index to metadata
                if chunk_metadata:
                    chunk_metadata["chunk_index"] = i
            
            # Process chunk
            self.process_text(chunk, chunk_metadata)
    
    def post_process(self, min_occurrences: int = 2) -> None:
        """
        Post-process the knowledge graph to improve quality.
        
        Args:
            min_occurrences: Minimum number of occurrences for an entity to keep
        """
        # Count entity occurrences
        entity_counts = defaultdict(int)
        for entity_id in self.knowledge_graph.entities:
            entity_counts[entity_id] += 1
        
        # Remove infrequent entities
        to_remove = []
        for entity_id, count in entity_counts.items():
            if count < min_occurrences:
                to_remove.append(entity_id)
        
        # Skip for now until we implement delete methods in KnowledgeGraph
        logger.info(f"Post-processing found {len(to_remove)} infrequent entities to be removed")
    
    def get_knowledge_graph(self) -> KnowledgeGraph:
        """
        Get the constructed knowledge graph.
        
        Returns:
            KnowledgeGraph instance
        """
        return self.knowledge_graph
    
    def reset(self) -> None:
        """
        Reset the extractor, clearing the knowledge graph.
        """
        self.knowledge_graph = KnowledgeGraph()
    
    def _map_entity_type(self, spacy_type: str) -> str:
        """
        Map spaCy entity types to simplified types.
        
        Args:
            spacy_type: spaCy entity type
            
        Returns:
            Simplified entity type
        """
        type_mapping = {
            "PERSON": "Person",
            "ORG": "Organization",
            "GPE": "Location",
            "LOC": "Location",
            "PRODUCT": "Product",
            "EVENT": "Event",
            "WORK_OF_ART": "WorkOfArt",
            "FAC": "Facility",
            "NORP": "Group",
            "LAW": "Law",
            "DATE": "Date",
            "TIME": "Time",
            "MONEY": "Money",
            "QUANTITY": "Quantity",
            "CARDINAL": "Number",
            "ORDINAL": "Number",
            "PERCENT": "Percent",
            "LANGUAGE": "Language"
        }
        
        return type_mapping.get(spacy_type, "Other")