import logging
import re
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class TemplateSelector:
    """Improved template selector that handles errors better."""
    
    def __init__(self, templates=None):
        self.templates = templates or {}
        self._load_default_templates()
        self.selected_template_name = "default"
    
    def _load_default_templates(self):
        """Load default templates if none were provided."""
        if not self.templates:
            try:
                from config import (
                    DEFAULT_PROMPT_TEMPLATE,
                    ENHANCED_PROMPT_TEMPLATE,
                    DOCUMENT_SUMMARY_TEMPLATE,
                    DOCUMENT_COMPARE_TEMPLATE,
                    TECHNICAL_DOCUMENT_TEMPLATE
                )
                
                self.templates = {
                    "default": DEFAULT_PROMPT_TEMPLATE,
                    "enhanced": ENHANCED_PROMPT_TEMPLATE,
                    "summary": DOCUMENT_SUMMARY_TEMPLATE,
                    "compare": DOCUMENT_COMPARE_TEMPLATE,
                    "technical": TECHNICAL_DOCUMENT_TEMPLATE
                }
            except (ImportError, AttributeError):
                self.templates = {"default": ""}
                logger.warning("Could not load templates from config")
    
    def select_template(self, query: str, context: str, metadata=None) -> str:
        """Select the best template based on query and context."""
        try:
            # Default to enhanced template
            self.selected_template_name = "enhanced"
            
            # Check for summary requests
            summary_patterns = [
                r'summarize', r'summary', r'overview', r'main points',
                r'what is this (document|text|content) about',
                r'describe the (document|text|content)',
                r'key (points|findings|ideas|concepts)',
                r'tldr', r'gist', r'brief description'
            ]
            
            if any(re.search(pattern, query.lower()) for pattern in summary_patterns):
                self.selected_template_name = "summary"
                logger.info(f"Selected summary template based on query")
            
            # Check for comparison requests
            compare_patterns = [
                r'compare', r'contrast', r'difference', r'similarities',
                r'how does .+ differ from', r'what is the relationship between',
                r'versus', r'vs\.', r'similarities and differences'
            ]
            
            if any(re.search(pattern, query.lower()) for pattern in compare_patterns):
                self.selected_template_name = "compare"
                logger.info(f"Selected comparison template based on query")
            
            # Make sure we have a valid template
            if self.selected_template_name not in self.templates:
                self.selected_template_name = "default"
            
            logger.info(f"Selected template: {self.selected_template_name}")
            return self.templates[self.selected_template_name]
            
        except Exception as e:
            logger.warning(f"Error in template selection: {e}")
            self.selected_template_name = "default"
            return self.templates.get("default", "")