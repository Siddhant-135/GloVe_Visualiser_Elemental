import numpy as np
from typing import List, Tuple, Optional
from embedding_manager import EmbeddingManager

class VectorArithmetic:
    """Handles vector arithmetic operations on word embeddings."""
    
    def __init__(self, embedding_manager: EmbeddingManager):
        self.embedding_manager = embedding_manager
        
    def perform_arithmetic(self, operations: List[Tuple[str, str]]) -> Tuple[np.ndarray, List[str]]:
        """
        Perform vector arithmetic based on a list of operations.
        
        Args:
            operations: List of (operation, word) tuples where operation is '+' or '-'
            
        Returns:
            Tuple of (result_vector, missing_words)
        """
        if not operations:
            raise ValueError("No operations provided")
        
        result_vector = None
        missing_words = []
        
        for operation, word in operations:
            # Get word vector with case-insensitive lookup
            word_vector = self.embedding_manager.get_word_vector(word)
            
            if word_vector is None:
                missing_words.append(word)
                continue
            
            if result_vector is None:
                # First word - initialize result vector
                if operation == '+':
                    result_vector = word_vector.copy()
                elif operation == '-':
                    result_vector = -word_vector.copy()
            else:
                # Subsequent words - apply operation
                if operation == '+':
                    result_vector += word_vector
                elif operation == '-':
                    result_vector -= word_vector
        
        if result_vector is None:
            # All words were missing
            result_vector = np.zeros(self.embedding_manager.embedding_dim)
        
        return result_vector, missing_words
    
    def find_similar_words(self, vector: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        """
        Find words most similar to a given vector.
        
        Args:
            vector: The query vector
            k: Number of similar words to return
            
        Returns:
            List of (word, similarity_score) tuples
        """
        return self.embedding_manager.find_most_similar(vector, k)
    
    def compute_analogy(self, word_a: str, word_b: str, word_c: str, k: int = 5) -> Tuple[List[Tuple[str, float]], List[str]]:
        """
        Compute analogy: word_a is to word_b as word_c is to ?
        This is equivalent to: word_b - word_a + word_c
        
        Args:
            word_a: First word in analogy
            word_b: Second word in analogy
            word_c: Third word in analogy
            k: Number of results to return
            
        Returns:
            Tuple of (similar_words, missing_words)
        """
        operations = [
            ('+', word_b),
            ('-', word_a),
            ('+', word_c)
        ]
        
        result_vector, missing_words = self.perform_arithmetic(operations)
        
        if missing_words:
            return [], missing_words
        
        similar_words = self.find_similar_words(result_vector, k)
        
        # Filter out the input words from results
        input_words = {word_a.lower(), word_b.lower(), word_c.lower()}
        filtered_results = [
            (word, score) for word, score in similar_words 
            if word.lower() not in input_words
        ]
        
        return filtered_results, missing_words
    
    def validate_words(self, words: List[str]) -> Tuple[List[str], List[str]]:
        """
        Validate a list of words against the vocabulary.
        
        Args:
            words: List of words to validate
            
        Returns:
            Tuple of (valid_words, invalid_words)
        """
        valid_words = []
        invalid_words = []
        
        for word in words:
            if self.embedding_manager.has_word(word):
                valid_words.append(word)
            else:
                invalid_words.append(word)
        
        return valid_words, invalid_words
    
    def get_word_similarity(self, word1: str, word2: str) -> Optional[float]:
        """
        Calculate cosine similarity between two words.
        
        Args:
            word1: First word
            word2: Second word
            
        Returns:
            Cosine similarity or None if either word is not found
        """
        vec1 = self.embedding_manager.get_word_vector(word1)
        vec2 = self.embedding_manager.get_word_vector(word2)
        
        if vec1 is None or vec2 is None:
            return None
        
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
