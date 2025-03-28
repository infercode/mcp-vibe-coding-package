from typing import Any, Dict, List, Optional

import numpy as np
from src.embedding_manager import LiteLLMEmbeddingManager

class EmbeddingAdapter:
    """
    Adapter for embedding generation functionality.
    Provides a clean interface for generating and utilizing embeddings.
    """
    
    def __init__(self, embedding_manager: Optional[LiteLLMEmbeddingManager] = None, logger=None):
        """
        Initialize the embedding adapter.
        
        Args:
            embedding_manager: The embedding manager to use, or None to create a new one
            logger: Optional logger instance
        """
        self.embedding_manager = embedding_manager
        self.logger = logger
        self._initialized = embedding_manager is not None
        
        if embedding_manager:
            self.model_name = getattr(embedding_manager, "model_name", "unknown")
            self.embedding_dim = getattr(embedding_manager, "embedding_dim", 1536)  # Default OpenAI dimension
        else:
            self.model_name = "unknown"
            self.embedding_dim = 1536  # Default OpenAI dimension
    
    def init_embedding_manager(self, api_key: Optional[str] = None, 
                             model_name: Optional[str] = None) -> bool:
        """
        Initialize the embedding manager if not already initialized.
        
        Args:
            api_key: Optional API key for the embedding model
            model_name: Optional model name for the embedding model
            
        Returns:
            True if initialized successfully, False otherwise
        """
        if self._initialized and self.embedding_manager:
            return True
        
        try:
            self.embedding_manager = LiteLLMEmbeddingManager(api_key=api_key, model_name=model_name)
            self._initialized = True
            
            # Update properties from the new manager
            if self.embedding_manager:
                self.model_name = getattr(self.embedding_manager, "model_name", "unknown")
                self.embedding_dim = getattr(self.embedding_manager, "embedding_dim", 1536)
            
            if self.logger:
                self.logger.info(f"Initialized embedding manager with model: {self.model_name}")
            
            return True
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to initialize embedding manager: {str(e)}")
            return False
    
    def ensure_initialized(self) -> bool:
        """
        Ensure that the embedding manager is initialized.
        
        Returns:
            True if initialized, False otherwise
        """
        if not self._initialized or not self.embedding_manager:
            # Try to initialize with default settings
            return self.init_embedding_manager()
        return True
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """
        Get embedding for text.
        
        Args:
            text: The text to get embedding for
            
        Returns:
            List of float values representing the embedding vector, or None if failed
        """
        if not self.ensure_initialized():
            if self.logger:
                self.logger.error("Cannot generate embedding: embedding manager not initialized")
            return None
        
        try:
            if not text:
                return None
            
            # Clean text before embedding
            cleaned_text = self._clean_text(text)
            if not cleaned_text:
                return None
            
            # Generate embedding
            embedding = self.embedding_manager.get_embedding(cleaned_text)
            
            if embedding is None:
                if self.logger:
                    self.logger.warning(f"Failed to generate embedding for text: {text[:100]}...")
                return None
            
            # Validate embedding
            if isinstance(embedding, list) and len(embedding) > 0:
                return embedding
            
            if self.logger:
                self.logger.warning(f"Invalid embedding generated for text: {text[:100]}...")
            return None
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error generating embedding: {str(e)}")
            return None
    
    def get_embeddings_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """
        Get embeddings for multiple texts in batch.
        
        Args:
            texts: List of texts to get embeddings for
            
        Returns:
            List of embeddings (each is list of floats), with None for failed items
        """
        if not self.ensure_initialized():
            if self.logger:
                self.logger.error("Cannot generate embeddings: embedding manager not initialized")
            return [None] * len(texts)
        
        try:
            results = []
            
            for text in texts:
                results.append(self.get_embedding(text))
            
            return results
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error generating batch embeddings: {str(e)}")
            return [None] * len(texts)
    
    def similarity_score(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        try:
            if not embedding1 or not embedding2:
                return 0.0
            
            # Convert to numpy arrays for efficient calculation
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            
            # Normalize to range 0-1
            return max(0.0, min(1.0, float(similarity)))
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error calculating similarity: {str(e)}")
            return 0.0
    
    def find_most_similar(self, query_embedding: List[float], 
                        candidate_embeddings: List[List[float]],
                        top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find the most similar embeddings from a list of candidates.
        
        Args:
            query_embedding: The embedding to compare against
            candidate_embeddings: List of candidate embeddings
            top_k: Number of top results to return
            
        Returns:
            List of dictionaries with index and similarity score
        """
        try:
            if not query_embedding or not candidate_embeddings:
                return []
            
            # Calculate similarities for all candidates
            similarities = []
            for i, candidate in enumerate(candidate_embeddings):
                if candidate:
                    score = self.similarity_score(query_embedding, candidate)
                    similarities.append({"index": i, "score": score})
            
            # Sort by similarity score
            similarities.sort(key=lambda x: x["score"], reverse=True)
            
            # Return top-k results
            return similarities[:top_k]
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error finding most similar embeddings: {str(e)}")
            return []
    
    def _clean_text(self, text: str) -> str:
        """
        Clean text before generating embeddings.
        
        Args:
            text: The text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Basic cleaning (can be extended with more sophisticated processing)
        cleaned = text.strip()
        
        # Remove excessive whitespace
        cleaned = " ".join(cleaned.split())
        
        return cleaned 