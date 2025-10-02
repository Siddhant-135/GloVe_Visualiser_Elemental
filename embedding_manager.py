import numpy as np
import faiss
from typing import Dict, List, Tuple, Optional

class EmbeddingManager:
    """Manages word embeddings and provides efficient similarity search using Faiss."""
    
    def __init__(self):
        self.embeddings = None
        self.word_to_index = {}
        self.index_to_word = {}
        self.embedding_dim = 0
        self.faiss_index = None
        
    def load_embeddings_from_file(self, file_path: str) -> None:
        """
        Load embeddings from a GloVe format file.
        
        Args:
            file_path: Path to the embedding file
        """
        words = []
        vectors = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split()
                if len(parts) < 2:
                    continue
                    
                word = parts[0]
                try:
                    # Convert the rest to float values
                    vector = [float(x) for x in parts[1:]]
                    
                    # Set embedding dimension from first word
                    if self.embedding_dim == 0:
                        self.embedding_dim = len(vector)
                    elif len(vector) != self.embedding_dim:
                        # Skip words with inconsistent dimensions
                        continue
                        
                    words.append(word)
                    vectors.append(vector)
                    
                except ValueError:
                    # Skip lines that can't be parsed
                    continue
        
        if not words:
            raise ValueError("No valid embeddings found in file")
        
        # Convert to numpy array
        self.embeddings = np.array(vectors, dtype=np.float32)
        
        # Create word mappings
        self.word_to_index = {word: i for i, word in enumerate(words)}
        self.index_to_word = {i: word for i, word in enumerate(words)}
        
        # Initialize Faiss index for efficient similarity search
        self._build_faiss_index()
        
    def _build_faiss_index(self) -> None:
        """Build Faiss index for efficient similarity search."""
        if self.embeddings is None:
            raise ValueError("Embeddings not loaded")
            
        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        normalized_embeddings = self.embeddings / norms
        
        # Create Faiss index (using inner product for cosine similarity with normalized vectors)
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        self.faiss_index.add(normalized_embeddings)
        
    def get_word_vector(self, word: str) -> Optional[np.ndarray]:
        """
        Get the embedding vector for a word.
        
        Args:
            word: The word to get embedding for
            
        Returns:
            Embedding vector or None if word not found
        """
        if self.embeddings is None:
            return None
            
        # Try exact match first
        if word in self.word_to_index:
            return self.embeddings[self.word_to_index[word]].copy()
        
        # Try lowercase
        word_lower = word.lower()
        if word_lower in self.word_to_index:
            return self.embeddings[self.word_to_index[word_lower]].copy()
        
        # Try uppercase
        word_upper = word.upper()
        if word_upper in self.word_to_index:
            return self.embeddings[self.word_to_index[word_upper]].copy()
        
        # Try title case
        word_title = word.title()
        if word_title in self.word_to_index:
            return self.embeddings[self.word_to_index[word_title]].copy()
        
        return None
    
    def find_most_similar(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        """
        Find the k most similar words to a query vector.
        
        Args:
            query_vector: The query vector
            k: Number of similar words to return
            
        Returns:
            List of (word, similarity_score) tuples
        """
        if self.faiss_index is None:
            raise ValueError("Faiss index not built")
        
        # Normalize query vector
        query_norm = query_vector / np.linalg.norm(query_vector)
        query_norm = query_norm.reshape(1, -1).astype(np.float32)
        
        # Search for similar vectors
        D, I = self.faiss_index.search(query_norm, k)
        similarities = D[0]
        indices = I[0]
        
        results = []
        for i in range(len(indices)):
            word_idx = int(indices[i])
            similarity = float(similarities[i])
            word = self.index_to_word[word_idx]
            results.append((word, similarity))
        
        return results
    
    def has_word(self, word: str) -> bool:
        """
        Check if a word exists in the vocabulary.
        
        Args:
            word: The word to check
            
        Returns:
            True if word exists, False otherwise
        """
        return self.get_word_vector(word) is not None
    
    def get_vocabulary_sample(self, n: int = 10) -> List[str]:
        """
        Get a sample of words from the vocabulary.
        
        Args:
            n: Number of words to return
            
        Returns:
            List of sample words
        """
        words = list(self.word_to_index.keys())
        return words[:min(n, len(words))]
