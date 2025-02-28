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
        # Extract query from prompt
        query_part = prompt.split("Question:")[-1].strip().split("\n")[0].strip() if "Question:" in prompt else ""
        
        # Extract context from prompt
        context_part = prompt.split("Context:")[1].split("Question:")[0].strip() if "Context:" in prompt else ""
        
        # Create a simple response
        if not context_part:
            return "I don't have enough information to answer this question."
        
        # Count the number of document chunks
        num_chunks = context_part.count("Document ")
        
        response = f"Based on the {num_chunks} document chunks I found, "
        
        # Extract first sentence from each document to summarize
        docs = context_part.split("Document ")[1:]
        if docs:
            response += "here's what I found:\n\n"
            for i, doc in enumerate(docs[:3]):  # Limit to first 3 docs
                # Get the first couple of sentences
                sentences = doc.split(". ")[:2]
                snippet = ". ".join(sentences) + ("." if not sentences[-1].endswith(".") else "")
                response += f"- {snippet}\n"
                
            if query_part:
                response += f"\nThis information appears relevant to your question about {query_part}"
        else:
            response += "I couldn't find relevant information to answer your question."
        
        return response
    
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
        model_type: Type of model ('openai', 'huggingface', 'local')
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
            model_name = LLM_MODEL_NAME
        if api_key is None:
            api_key = LLM_API_KEY
    except (ImportError, AttributeError):
        # Use defaults if config not available
        if model_name is None:
            model_name = "gpt-3.5-turbo" if model_type == "openai" else None
    
    # Create model based on type
    if model_type.lower() == "openai":
        return OpenAIModel(model_name=model_name, api_key=api_key, **kwargs)
    elif model_type.lower() in ["huggingface", "hf"]:
        # Placeholder for future HuggingFace model implementation
        logger.warning("HuggingFace models not yet implemented, falling back to local model")
        return LocalLLM()
    else:
        # Default to local model
        return LocalLLM()