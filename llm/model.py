"""
Language model integration for generating responses.
"""

import os
import logging
from typing import Optional, Dict, Any, List
import json

# Configure logging
logger = logging.getLogger(__name__)


class BaseLLM:
    """Base class for language models."""
    
    def generate_response(self, prompt: str, max_tokens: int = 512) -> str:
        """
        Generate a response to a prompt.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated response
        """
        raise NotImplementedError("Subclasses must implement generate_response")


class OpenAIModel(BaseLLM):
    """OpenAI language model integration."""
    
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        temperature: float = 0.2
    ):
        """
        Initialize the OpenAI model.
        
        Args:
            model_name: Name of the OpenAI model
            api_key: OpenAI API key (defaults to OPENAI_API_KEY environment variable)
            temperature: Sampling temperature (0-1)
        """
        self.model_name = model_name
        self.temperature = temperature
        
        # Get API key
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("No OpenAI API key provided, model will not be functional")
        
        # Lazy import to avoid dependency issues
        try:
            import openai
            self.openai = openai
            self.openai.api_key = self.api_key
            self.available = True
        except ImportError:
            logger.warning("OpenAI package not installed, functionality will be limited")
            self.available = False
    
    def generate_response(self, prompt: str, max_tokens: int = 512) -> str:
        """
        Generate a response using the OpenAI API.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated response
        """
        if not self.available or not self.api_key:
            return "OpenAI integration not available. Please check your API key and dependencies."
        
        try:
            response = self.openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating response with OpenAI: {e}")
            return f"Error generating response: {str(e)}"
    
    def generate_openai_response(self, prompt: str, max_tokens: int = 512) -> str:
        """Alias for generate_response for compatibility."""
        return self.generate_response(prompt, max_tokens)


class LocalLLM(BaseLLM):
    """Simple local language model for when no external API is available."""
    
    def __init__(self):
        """Initialize the local LLM."""
        self.available = True
    
    def generate_response(self, prompt: str, max_tokens: int = 512) -> str:
        """
        Generate a simple response based on retrieved context.
        
        Args:
            prompt: Input prompt containing context and query
            max_tokens: Maximum response length (ignored in this simple implementation)
            
        Returns:
            Generated response summarizing the context
        """
        # Check for context and question in the prompt
        if "Context:" not in prompt or "Question:" not in prompt:
            logger.warning("Prompt missing required Context or Question sections")
            return "I don't have enough information to answer this question."
        
        try:
            # Extract context and question from prompt
            context_part = prompt.split("Context:")[1].split("Question:")[0].strip()
            question_part = prompt.split("Question:")[1].strip().split("\n")[0].strip()
            
            # Log what we extracted
            logger.info(f"Extracted context of length {len(context_part)} chars and question: {question_part}")
            
            # Check if context is empty or too short
            if not context_part or len(context_part) < 20:
                logger.warning("No context found in prompt or context too short")
                return "I don't have enough information to answer this question."
            
            # Count document chunks in context
            num_chunks = context_part.count("Document ")
            logger.info(f"Found {num_chunks} document chunks in context")
            
            # Extract content from document chunks
            chunks = []
            if num_chunks > 0:
                # Try to split by document markers
                doc_parts = context_part.split("Document ")
                # Skip the first empty part
                for part in doc_parts[1:]:
                    # Extract the document content
                    if "[Source:" in part:
                        # Extract text after the source indicator
                        content = part.split("[Source:")[1].split("]", 1)[1].strip()
                        chunks.append(content)
                    else:
                        # If no source indicator, just take the whole part
                        chunks.append(part.strip())
            else:
                # If no document markers, use the whole context
                chunks = [context_part]
            
            # Log extracted chunks
            logger.info(f"Extracted {len(chunks)} chunks from context")
            for i, chunk in enumerate(chunks[:2]):  # Log just first 2 chunks
                logger.info(f"Chunk {i+1} preview: {chunk[:100]}...")
            
            # If no usable chunks, return default response
            if not chunks or all(len(chunk.strip()) < 20 for chunk in chunks):
                logger.warning("No usable chunks found in context")
                return "I don't have enough information to answer this question."
            
            # For summarization questions
            if any(term in question_part.lower() for term in ['summarize', 'summary', 'overview', 'main points']):
                return self._generate_summary(chunks)
            
            # For comparison questions
            elif any(term in question_part.lower() for term in ['compare', 'contrast', 'difference between', 'similarities']):
                return self._generate_comparison(chunks, question_part)
            
            # For general questions
            else:
                return self._generate_answer(chunks, question_part)
                
        except Exception as e:
            logger.error(f"Error generating response in LocalLLM: {e}", exc_info=True)
            return "I encountered an error while processing your question."
    
    def _generate_summary(self, chunks: List[str]) -> str:
        """Generate a summary from document chunks."""
        # Create a simple summary based on the chunks
        if not chunks:
            return "There isn't enough information to provide a summary."
        
        response = "Based on the provided documents, here's a summary:\n\n"
        
        # Look for key sentences in each chunk
        for i, chunk in enumerate(chunks[:3]):  # Limit to first 3 chunks
            # Extract first sentence or two from each chunk
            sentences = chunk.split('.')
            summary_sentences = '. '.join(sentences[:min(3, len(sentences))]) + '.'
            
            response += f"- {summary_sentences}\n\n"
        
        return response
    
    def _generate_comparison(self, chunks: List[str], question: str) -> str:
        """Generate a comparison from document chunks."""
        if len(chunks) < 2:
            return "I don't have enough different document sections to make a comparison."
        
        response = "Comparing the information in the documents:\n\n"
        
        # Add information from first two chunks
        response += f"First section discusses: {chunks[0][:200]}...\n\n"
        response += f"Second section covers: {chunks[1][:200]}...\n\n"
        
        response += "In comparing these sections, they both relate to the topic but focus on different aspects."
        
        return response
    
    def _generate_answer(self, chunks: List[str], question: str) -> str:
        """Generate an answer to a question from document chunks."""
        # Try to find a chunk that might contain the answer
        best_chunk = None
        best_score = 0
        
        # Create list of important words from the question
        question_words = [w.lower() for w in question.split() if len(w) > 3]
        
        for chunk in chunks:
            chunk_lower = chunk.lower()
            # Count how many question words appear in this chunk
            score = sum(1 for word in question_words if word in chunk_lower)
            if score > best_score:
                best_score = score
                best_chunk = chunk
        
        if best_chunk and best_score > 0:
            # Use the best matching chunk for the answer
            sentences = best_chunk.split('.')
            # Get the first few sentences
            relevant_text = '. '.join(sentences[:min(4, len(sentences))]) + '.'
            
            return f"Based on the documents, {relevant_text}"
        else:
            return "I don't have enough information to answer that specific question."
    
    def generate_huggingface_response(self, prompt: str, max_tokens: int = 512) -> str:
        """Alias for generate_response for compatibility."""
        return self.generate_response(prompt, max_tokens)


# Factory function to create LLM instances
def create_llm(
    model_type: str = "local",
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> BaseLLM:
    """
    Create a language model instance.
    
    Args:
        model_type: Type of model ('openai', 'huggingface', 'serverless', 'local')
        model_name: Name of the model
        api_key: API key for the service
        **kwargs: Additional model parameters
        
    Returns:
        Language model instance
    """
    # Try to get config
    try:
        from config import LLM_MODEL_NAME, LLM_API_KEY
        
        # Set defaults from config if not provided
        if model_name is None:
            logger.info("Using default model name from config")
            model_name = LLM_MODEL_NAME
        if api_key is None:
            logger.info("Using default API key from config")
            api_key = LLM_API_KEY
    except (ImportError, AttributeError):
        # Use defaults if config not available
        if model_name is None:
            model_name = "gpt-3.5-turbo" if model_type == "openai" else None
    
    # Try to get API key from Streamlit secrets if available
    try:
        import streamlit as st
        if api_key is None and model_type == "openai":
            api_key = st.secrets.get("OPENAI_API_KEY")
        elif api_key is None and model_type in ["huggingface", "serverless"]:
            api_key = st.secrets.get("HUGGINGFACE_API_KEY")
    except:
        pass
        
    # Create model based on type
    if model_type.lower() == "openai":
        return OpenAIModel(model_name=model_name, api_key=api_key, **kwargs)
    elif model_type.lower() == "serverless":
        try:
            from .serverless_model import create_serverless_llm
            return create_serverless_llm(model_name=model_name, api_key=api_key, **kwargs)
        except ImportError:
            logger.warning("Serverless LLM not available, falling back to local model")
            return LocalLLM()
    elif model_type.lower() in ["huggingface", "hf"]:
        # Try to use serverless HuggingFace API
        try:
            from .serverless_model import HuggingFaceInferenceAPI
            return HuggingFaceInferenceAPI(model_name=model_name, api_key=api_key)
        except ImportError:
            # Fall back to local model
            logger.warning("HuggingFace API model not available, falling back to local model")
            return LocalLLM()
    else:
        # Default to local model
        return LocalLLM()