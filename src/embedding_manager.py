#!/usr/bin/env python3
from typing import List, Dict, Any, Optional, Union
import os
import json

from litellm import embedding

class LiteLLMEmbeddingManager:
    """Embedding manager using LiteLLM for multi-provider embedding support."""
    
    def __init__(self, logger):
        """Initialize the embedding manager."""
        self.logger = logger
        self.provider = "none"
        self.model = None
        self.dimensions = None
        self.embedding_enabled = False
        self.api_key = None
        self.api_base = None
        self.additional_params = {}
        
    def configure(self, config: Dict[str, Any]) -> Dict[str, str]:
        """
        Configure the embedding manager with the provided settings.
        
        Args:
            config: Dictionary containing embedding configuration
            
        Returns:
            Dictionary with status and message
        """
        try:
            if not config or "provider" not in config:
                return {"status": "error", "message": "Invalid configuration: missing provider"}
            
            # Extract base configuration
            self.provider = config.get("provider", "none")
            self.model = config.get("model")
            self.api_key = config.get("api_key")
            self.dimensions = config.get("dimensions")
            self.api_base = config.get("api_base")
            
            # Store additional parameters
            self.additional_params = config.get("additional_params", {})
            
            # Configure based on provider
            if self.provider == "none":
                self.embedding_enabled = False
                return {"status": "success", "message": "Embeddings disabled"}
            
            # Set embedding as enabled for all other providers
            self.embedding_enabled = True
            
            # Set up environment variables if needed
            if self.api_key:
                if self.provider == "openai":
                    os.environ["OPENAI_API_KEY"] = self.api_key
                elif self.provider in ["azure", "azure_openai"]:
                    os.environ["AZURE_API_KEY"] = self.api_key
                elif self.provider == "cohere":
                    os.environ["COHERE_API_KEY"] = self.api_key
                elif self.provider == "huggingface":
                    os.environ["HUGGINGFACE_API_KEY"] = self.api_key
                elif self.provider == "vertexai" or self.provider == "vertex_ai":
                    # Handle Vertex AI credentials
                    if "vertex_credentials_json" in self.additional_params:
                        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.additional_params["vertex_credentials_json"]
                elif self.provider == "gemini":
                    os.environ["GEMINI_API_KEY"] = self.api_key
                elif self.provider == "mistral":
                    os.environ["MISTRAL_API_KEY"] = self.api_key
                elif self.provider == "voyage":
                    os.environ["VOYAGE_API_KEY"] = self.api_key
            
            # Set up provider-specific API base URL if needed
            if self.api_base:
                if self.provider == "openai":
                    os.environ["OPENAI_API_BASE"] = self.api_base
                elif self.provider in ["azure", "azure_openai"]:
                    os.environ["AZURE_API_BASE"] = self.api_base
            
            # Handle Azure-specific settings
            if self.provider in ["azure", "azure_openai"]:
                # Set API version
                if "api_version" in self.additional_params:
                    os.environ["AZURE_API_VERSION"] = self.additional_params["api_version"]
                
                # Make sure deployment is properly set
                if "deployment" in self.additional_params:
                    # For LiteLLM, deployment name is passed in the API call, not as env var
                    # But we'll leave this for debugging purposes
                    self.logger.debug(f"Azure deployment: {self.additional_params['deployment']}")
                    
                # Format model name correctly for Azure
                if self.model and not self.model.startswith("azure/"):
                    # Azure model names should be prefixed with azure/
                    self.model = f"azure/{self.model}"
            else:
                # Construct model string for LiteLLM format (for non-Azure providers)
                if self.model:
                    if self.provider != "openai" and not self.model.startswith(f"{self.provider}/"):
                        self.model = f"{self.provider}/{self.model}"
            
            # Log complete configuration
            self.logger.info(f"Embedding manager configured with provider: {self.provider}", 
                            context={
                                "model": self.model,
                                "dimensions": self.dimensions,
                                "api_base_set": bool(self.api_base),
                                "api_key_set": bool(self.api_key),
                                "additional_params": {k: "..." if "key" in k.lower() else v for k, v in self.additional_params.items()}
                            })
            
            return {
                "status": "success", 
                "message": f"Embedding manager configured with provider: {self.provider}"
            }
            
        except Exception as e:
            self.logger.error(f"Error configuring embedding manager: {str(e)}")
            return {"status": "error", "message": f"Error configuring embedding manager: {str(e)}"}
    
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate an embedding for the given text.
        
        Args:
            text: The text to generate embedding for
            
        Returns:
            List of floats representing the embedding vector, or None if embedding fails
        """
        if not self.embedding_enabled or not self.model:
            return None
        
        try:
            # Prepare parameters
            params = {}
            
            # Add dimensions if specified
            if self.dimensions:
                params["dimensions"] = self.dimensions
            
            # Add api_base if specified
            if self.api_base:
                params["api_base"] = self.api_base
            
            # Add api_key if specified
            if self.api_key:
                params["api_key"] = self.api_key
            
            # Add additional provider-specific params
            params.update(self.additional_params)
            
            # Log embedding request (without sensitive data)
            self.logger.debug(f"Generating embedding for provider {self.provider} with model {self.model}", 
                             context={"text_length": len(text), "params_keys": list(params.keys())})
            
            # Generate embedding using LiteLLM
            response = embedding(
                model=self.model,
                input=[text],
                **params
            )
            
            # Handle async response if needed
            import asyncio
            if asyncio.iscoroutine(response):
                # Create event loop if needed
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    # Create new event loop if none is available
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                # Run the coroutine to get the actual response
                response = loop.run_until_complete(response)
            
            # Extract embedding from response
            if response and isinstance(response, dict) and "data" in response and len(response["data"]) > 0:
                embedding_data = response["data"][0]["embedding"]
                self.logger.debug(f"Embedding generated successfully with {len(embedding_data)} dimensions")
                return embedding_data
            
            self.logger.warn("Empty embedding response", context={"model": self.model, "response_type": type(response).__name__})
            return None
            
        except Exception as e:
            self.logger.error(f"Error generating embedding: {str(e)}", context={"text_length": len(text)})
            return None
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the current configuration of the embedding manager.
        
        Returns:
            Dictionary with the current configuration
        """
        config = {
            "provider": self.provider,
            "model": self.model,
            "dimensions": self.dimensions,
            "embedding_enabled": self.embedding_enabled
        }
        
        # Don't include sensitive information like API keys
        # Add any additional non-sensitive information
        if self.additional_params:
            safe_params = {k: v for k, v in self.additional_params.items() if "key" not in k.lower() and "password" not in k.lower()}
            if safe_params:
                config["additional_params"] = safe_params
        
        return config 