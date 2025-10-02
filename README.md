# Word Embeddings Explorer

## Overview

Word Embeddings Explorer is an interactive educational application built with Streamlit that allows users to explore and understand dense word embeddings through vector arithmetic operations. The application loads pre-trained word embeddings (GloVe format) and enables users to perform mathematical operations on word vectors to discover semantic relationships (e.g., "king - man + woman = queen"). The system uses efficient similarity search via Faiss to find the nearest neighbors of resulting vectors.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web application
- **Session State Management**: Uses Streamlit's session state to persist embedding manager, vector arithmetic engine, and user operations across reruns
- **File Upload Handling**: Implements file size validation with a 500MB maximum limit to prevent memory issues
- **Layout**: Wide layout configuration for better visualization of results

### Backend Architecture
- **Embedding Management**: 
  - `EmbeddingManager` class handles loading and storage of word embeddings
  - Supports GloVe text format (word followed by space-separated float values)
  - Maintains bidirectional mapping between words and indices for fast lookup
  - Implements case-insensitive word lookup for better user experience

- **Vector Operations**:
  - `VectorArithmetic` class encapsulates all vector arithmetic logic
  - Supports addition and subtraction operations on word vectors
  - Handles missing words gracefully by tracking them separately
  - Returns normalized result vectors for consistent similarity computations

- **Similarity Search**:
  - Leverages Faiss library for efficient nearest neighbor search
  - Indexes all embeddings for fast similarity queries
  - Enables finding words closest to arbitrary result vectors

### Data Storage
- **In-Memory Storage**: All embeddings stored in NumPy arrays for fast numerical operations
- **Index Structures**: 
  - Dictionary-based word-to-index and index-to-word mappings
  - Faiss index for vector similarity search
- **File Format**: Supports standard GloVe text format (space-separated values)
- **Default Dataset**: Includes GloVe 6B 50-dimensional embeddings as default

### Design Patterns
- **Separation of Concerns**: Clear division between embedding management, vector operations, and UI logic
- **State Management**: Centralized session state for maintaining application context
- **Error Handling**: Graceful degradation when words are not found in vocabulary
- **Lazy Initialization**: Components initialized only when needed (on user action)

## External Dependencies

### Core Libraries
- **Streamlit**: Web application framework for the interactive UI
- **NumPy**: Numerical operations and array manipulation for vector arithmetic
- **Pandas**: Data manipulation and display (imported but usage not shown in provided code)
- **Faiss**: Facebook's library for efficient similarity search and clustering of dense vectors

### Data Dependencies
- **Pre-trained Embeddings**: GloVe (Global Vectors for Word Representation) embeddings
  - Default: 6B tokens, 50-dimensional vectors
  - Format: Plain text with space-separated values
  - Location: `attached_assets/glove.6B.50d_1759386912764.txt`

### File System
- Uses local file system for loading pre-trained embedding files
- Supports custom file uploads for user-provided embeddings
