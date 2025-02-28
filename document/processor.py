"""
Document processing utilities for text extraction and chunking.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
import uuid

# Configure logging
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Class to handle document processing, chunking, and text extraction.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Maximum size of text chunks in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def process_file(
        self,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Process a document file: extract text and chunk it.
        
        Args:
            file_path: Path to the document file
            metadata: Optional metadata about the document
            
        Returns:
            Tuple of (list of text chunks, list of metadata dictionaries)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        # Extract text from document
        logger.info(f"Processing document: {file_path}")
        text = self._extract_text(file_path)
        
        if not text:
            logger.warning(f"No text could be extracted from {file_path}")
            return [], []
        
        # Create base metadata if not provided
        base_metadata = {"source": os.path.basename(file_path)}
        if metadata:
            base_metadata.update(metadata)
        
        # Chunk the document
        chunks = self._chunk_text(text, self.chunk_size, self.chunk_overlap)
        logger.info(f"Created {len(chunks)} chunks from document")
        
        # Create chunk-specific metadata
        chunk_metadata = []
        for i, _ in enumerate(chunks):
            metadata_item = {
                **base_metadata,
                "chunk_id": i,
                "total_chunks": len(chunks),
                "document_id": str(uuid.uuid4())  # Unique ID for tracking
            }
            chunk_metadata.append(metadata_item)
        
        return chunks, chunk_metadata
    
    def _extract_text(self, file_path: str) -> str:
        """
        Extract text from a document file based on its extension.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Extracted text
        """
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        if ext == '.pdf':
            return self._extract_text_from_pdf(file_path)
        elif ext == '.txt':
            return self._extract_text_from_txt(file_path)
        elif ext == '.md':
            return self._extract_text_from_txt(file_path)
        elif ext == '.docx':
            return self._extract_text_from_docx(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
    
    def _extract_text_from_pdf(self, file_path: str) -> str:
        """
        Extract text from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text
        """
        try:
            import PyPDF2
        except ImportError:
            raise ImportError(
                "PyPDF2 is not installed. "
                "Please install it with `pip install PyPDF2`."
            )
        
        text = ""
        try:
            with open(file_path, "rb") as f:
                pdf_reader = PyPDF2.PdfReader(f)
                num_pages = len(pdf_reader.pages)
                logger.info(f"PDF has {num_pages} pages")
                
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
        except Exception as e:
            logger.error(f"Error reading PDF file {file_path}: {e}")
        
        logger.info(f"Extracted {len(text)} characters from PDF")
        return text
    
    def _extract_text_from_txt(self, file_path: str) -> str:
        """
        Extract text from a plain text file.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            Extracted text
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            logger.info(f"Extracted {len(text)} characters from text file")
            return text
        except Exception as e:
            logger.error(f"Error reading text file {file_path}: {e}")
            return ""
    
    def _extract_text_from_docx(self, file_path: str) -> str:
        """
        Extract text from a DOCX file.
        
        Args:
            file_path: Path to the DOCX file
            
        Returns:
            Extracted text
        """
        try:
            import docx
        except ImportError:
            raise ImportError(
                "python-docx is not installed. "
                "Please install it with `pip install python-docx`."
            )
        
        try:
            doc = docx.Document(file_path)
            text = "\n\n".join([paragraph.text for paragraph in doc.paragraphs if paragraph.text])
            
            logger.info(f"Extracted {len(text)} characters from DOCX")
            return text
        except Exception as e:
            logger.error(f"Error reading DOCX file {file_path}: {e}")
            return ""
    
    @staticmethod
    def _chunk_text(
        text: str,
        chunk_size: int = 1000,
        overlap: int = 200
    ) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: The text to chunk
            chunk_size: Maximum chunk size in characters
            overlap: Overlap between chunks in characters
            
        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []
            
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            # Define the initial chunk end
            end = min(start + chunk_size, text_len)
            
            # Try to find a natural break point if not at the end of text
            if end < text_len:
                # Look for paragraph break
                next_para = text.find('\n\n', end - overlap, end + 100)
                if next_para != -1:
                    end = next_para + 2
                else:
                    # Look for sentence break
                    for punct in ['. ', '! ', '? ', '.\n', '!\n', '?\n']:
                        next_sent = text.find(punct, end - overlap, end + 100)
                        if next_sent != -1:
                            end = next_sent + len(punct)
                            break
            
            # Extract the chunk
            chunk = text[start:end].strip()
            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)
            
            # Move to next chunk with overlap
            start = max(end - overlap, start + 1)
            
        return chunks
