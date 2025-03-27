#!/usr/bin/env python3
from langchain.embeddings.base import Embeddings
from typing import List, Dict, Any, Optional
import litellm
import os
import asyncio

class LiteLLMEmbeddings(Embeddings):
    """LangChain compatible wrapper for LiteLLM embeddings."""
    
    def __init__(
        self, 
        model: str, 
        api_key: Optional[str] = None,
        dimensions: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize with model name and optional parameters.
        
        Args:
            model: The name of the embedding model to use, 
                  following LiteLLM's provider/model format (e.g., "openai/text-embedding-3-small")
            api_key: The API key for the selected provider
            dimensions: Optional number of dimensions for the embeddings
            **kwargs: Additional arguments to pass to the LiteLLM embedding function
        """
        self.model = model
        self.dimensions = dimensions
        self.kwargs = kwargs
        
        # Handle API key setup
        self.api_key = api_key
        if api_key:
            # Set appropriate environment variable based on provider
            provider = model.split('/')[0] if '/' in model else model
            if provider == "openai":
                os.environ["OPENAI_API_KEY"] = api_key
            elif provider == "azure":
                os.environ["AZURE_API_KEY"] = api_key
            elif provider == "cohere":
                os.environ["COHERE_API_KEY"] = api_key
            elif provider == "huggingface":
                os.environ["HUGGINGFACE_API_KEY"] = api_key
            elif provider in ["vertexai", "vertex_ai"]:
                if "vertex_credentials_json" in kwargs:
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = kwargs["vertex_credentials_json"]
            elif provider == "gemini":
                os.environ["GEMINI_API_KEY"] = api_key
            elif provider == "mistral":
                os.environ["MISTRAL_API_KEY"] = api_key
            elif provider == "voyage":
                os.environ["VOYAGE_API_KEY"] = api_key
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents using LiteLLM.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        # Prepare parameters
        params = {}
        if self.api_key:
            params["api_key"] = self.api_key
        if self.dimensions:
            params["dimensions"] = self.dimensions
            
        # Add any additional parameters
        params.update(self.kwargs)
        
        # Call LiteLLM's embedding function in batch
        try:
            # Handle both synchronous and asynchronous cases
            response = litellm.embedding(
                model=self.model,
                input=texts,
                **params
            )
            
            # If response is a coroutine, run it in an event loop
            if asyncio.iscoroutine(response):
                # Create a new event loop if needed
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                response = loop.run_until_complete(response)
            
            # Extract embeddings from response
            embeddings = [item["embedding"] for item in response["data"]]
            return embeddings
            
        except Exception as e:
            raise ValueError(f"Error generating embeddings with LiteLLM: {str(e)}")
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text using LiteLLM.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        # Call embed_documents and return the first result
        return self.embed_documents([text])[0] 