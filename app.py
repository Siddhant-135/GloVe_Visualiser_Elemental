import streamlit as st
import numpy as np
import pandas as pd
from embedding_manager import EmbeddingManager
from vector_operations import VectorArithmetic
import os

st.set_page_config(
    page_title="Word Embeddings Explorer",
    page_icon="🧮",
    layout="wide"
)

if 'embedding_manager' not in st.session_state:
    st.session_state.embedding_manager = None
if 'vector_arithmetic' not in st.session_state:
    st.session_state.vector_arithmetic = None
if 'operations' not in st.session_state:
    st.session_state.operations = []

MAX_FILE_SIZE_MB = 500

def check_file_size(file):
    """Check if uploaded file is too large."""
    file_size_mb = file.size / (1024 * 1024)
    return file_size_mb, file_size_mb <= MAX_FILE_SIZE_MB

def main():
    st.title("🧮 Word Embeddings Explorer")
    st.markdown("### Learn about dense embeddings through interactive vector arithmetic")
    
    with st.sidebar:
        st.header("Configuration")
        
        st.subheader("📁 Load Embeddings")
        
        default_file_path = "attached_assets/glove.6B.50d_1759386912764.txt"
        if os.path.exists(default_file_path):
            st.info("GloVe 50d embeddings available")
            if st.button("Load Default Embeddings", type="primary"):
                with st.spinner("Loading embeddings..."):
                    try:
                        st.session_state.embedding_manager = EmbeddingManager()
                        st.session_state.embedding_manager.load_embeddings_from_file(default_file_path)
                        st.session_state.vector_arithmetic = VectorArithmetic(st.session_state.embedding_manager)
                        st.success(f"Loaded {len(st.session_state.embedding_manager.word_to_index)} words!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error loading embeddings: {str(e)}")
        
        st.markdown("---")
        st.subheader("📤 Upload Your Own Embeddings")
        uploaded_file = st.file_uploader(
            "Upload GloVe format file",
            type=['txt'],
            help="Upload a text file with word embeddings in GloVe format"
        )
        
        if uploaded_file is not None:
            file_size_mb, is_valid_size = check_file_size(uploaded_file)
            
            if not is_valid_size:
                st.error(f"⚠️ File is too large ({file_size_mb:.1f} MB). Maximum allowed size is {MAX_FILE_SIZE_MB} MB.")
            else:
                st.info(f"File size: {file_size_mb:.1f} MB")
                if st.button("Load Uploaded File"):
                    with st.spinner("Processing embeddings..."):
                        temp_path = f"temp_{uploaded_file.name}"
                        try:
                            with open(temp_path, "wb") as f:
                                f.write(uploaded_file.getvalue())
                            
                            st.session_state.embedding_manager = EmbeddingManager()
                            st.session_state.embedding_manager.load_embeddings_from_file(temp_path)
                            st.session_state.vector_arithmetic = VectorArithmetic(st.session_state.embedding_manager)
                            
                            if os.path.exists(temp_path):
                                os.remove(temp_path)
                            
                            st.success(f"Loaded {len(st.session_state.embedding_manager.word_to_index)} words with {st.session_state.embedding_manager.embedding_dim} dimensions!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error loading embeddings: {str(e)}")
                            if os.path.exists(temp_path):
                                os.remove(temp_path)
        
        if st.session_state.embedding_manager:
            st.markdown("---")
            st.subheader("📊 Embedding Info")
            st.metric("Vocabulary Size", len(st.session_state.embedding_manager.word_to_index))
            st.metric("Embedding Dimension", st.session_state.embedding_manager.embedding_dim)
    
    st.markdown("---")
    st.markdown("### 🎓 About Word Embeddings")
    st.markdown("""
    Word embeddings are dense vector representations of words that capture semantic relationships.
    Similar words have similar vectors, enabling mathematical operations on meaning:
    
    - **king** - **man** + **woman** ≈ **queen**
    - **Paris** - **France** + **Italy** ≈ **Rome**
    
    This tool lets you explore these relationships interactively using vector arithmetic!
    """)
    
    st.markdown("### 📝 How to Use")
    st.markdown("""
    1. **Load embeddings**: Click 'Load Default Embeddings' or upload your own file in the sidebar
    2. **Build operations**: Add up to 3 words with + or - buttons
    3. **Calculate**: Click the Calculate button to see results
    4. **Explore**: See the top matching words with similarity scores!
    """)
    
    if st.session_state.embedding_manager is None:
        st.info("👆 Please load embeddings from the sidebar to get started!")
        return
    
    st.markdown("---")
    st.header("🔢 Vector Arithmetic")
    
    st.markdown("**Build your operation (up to 3 words):**")
    
    if 'op1_state' not in st.session_state:
        st.session_state.op1_state = "+"
    if 'op2_state' not in st.session_state:
        st.session_state.op2_state = "+"
    if 'op3_state' not in st.session_state:
        st.session_state.op3_state = "+"
    
    cols = st.columns([0.5, 0.5, 2, 0.5, 0.5, 2, 0.5, 0.5, 2, 1.5])
    
    if cols[0].button("+", key="op1_plus", type="primary" if st.session_state.op1_state == "+" else "secondary", use_container_width=True):
        st.session_state.op1_state = "+"
    if cols[1].button("-", key="op1_minus", type="primary" if st.session_state.op1_state == "-" else "secondary", use_container_width=True):
        st.session_state.op1_state = "-"
    
    word1 = cols[2].text_input("Word 1", key="word_1", placeholder="Enter first word...", label_visibility="collapsed")
    
    if cols[3].button("+", key="op2_plus", type="primary" if st.session_state.op2_state == "+" else "secondary", use_container_width=True):
        st.session_state.op2_state = "+"
    if cols[4].button("-", key="op2_minus", type="primary" if st.session_state.op2_state == "-" else "secondary", use_container_width=True):
        st.session_state.op2_state = "-"
    
    word2 = cols[5].text_input("Word 2", key="word_2", placeholder="Enter second word...", label_visibility="collapsed")
    
    if cols[6].button("+", key="op3_plus", type="primary" if st.session_state.op3_state == "+" else "secondary", use_container_width=True):
        st.session_state.op3_state = "+"
    if cols[7].button("-", key="op3_minus", type="primary" if st.session_state.op3_state == "-" else "secondary", use_container_width=True):
        st.session_state.op3_state = "-"
    
    word3 = cols[8].text_input("Word 3", key="word_3", placeholder="Enter third word...", label_visibility="collapsed")
    
    k = st.slider("Number of similar words to find (k)", 1, 10, 5, key="k_slider")
    
    if cols[9].button("🚀 Calculate", type="primary", use_container_width=True):
        operations = []
        input_words = []
        
        if word1.strip():
            operations.append((st.session_state.op1_state, word1.strip().lower()))
            input_words.append(word1.strip().lower())
        
        if word2.strip():
            operations.append((st.session_state.op2_state, word2.strip().lower()))
            input_words.append(word2.strip().lower())
        
        if word3.strip():
            operations.append((st.session_state.op3_state, word3.strip().lower()))
            input_words.append(word3.strip().lower())
        
        if len(operations) > 0:
            with st.spinner("Computing vector arithmetic..."):
                try:
                    result_vector, missing_words = st.session_state.vector_arithmetic.perform_arithmetic(operations)
                    
                    if missing_words:
                        error_msg = "❌ **The following word"
                        if len(missing_words) > 1:
                            error_msg += "s are"
                        else:
                            error_msg += " is"
                        error_msg += " not in my vocabulary:**\n\n"
                        for word in missing_words:
                            error_msg += f"- `{word}`\n"
                        st.error(error_msg)
                    else:
                        similar_words = st.session_state.vector_arithmetic.find_similar_words(result_vector, k + 10)
                        
                        filtered_similar = [(w, s) for w, s in similar_words if w.lower() not in input_words][:k]
                        
                        if filtered_similar:
                            top_word, top_similarity = filtered_similar[0]
                            
                            st.markdown("---")
                            st.header("📊 Results")
                            
                            operation_display = ""
                            for i, (op, word) in enumerate(operations):
                                if i == 0:
                                    operation_display = f"**{word}**"
                                else:
                                    operation_display += f" {op} **{word}**"
                            
                            st.markdown(f"### {operation_display} ≈ **{top_word}** (similarity: {top_similarity:.4f})")
                            
                            st.markdown("**Resultant Vector (first 10 dimensions):**")
                            vector_values = [f"{val:.3f}" for val in result_vector[:10]]
                            vector_str = ", ".join(vector_values)
                            if len(result_vector) > 10:
                                vector_str += ", \\ldots"
                            st.latex(r"\mathbf{v} = \langle " + vector_str + r" \rangle")
                            
                            st.markdown("---")
                            st.subheader(f"Top {k} Closest Matches")
                            
                            col_list, col_chart = st.columns(2)
                            
                            with col_list:
                                st.markdown("**Ranked Results:**")
                                for i, (word, similarity) in enumerate(filtered_similar):
                                    similarity_percentage = similarity * 100
                                    st.markdown(f"{i+1}. **{word}** — {similarity_percentage:.2f}% (similarity: {similarity:.4f})")
                            
                            with col_chart:
                                st.markdown("**Similarity Chart:**")
                                chart_data = pd.DataFrame({
                                    'Word': [word for word, _ in filtered_similar],
                                    'Similarity': [sim for _, sim in filtered_similar]
                                })
                                st.bar_chart(chart_data.set_index('Word'))
                        
                except Exception as e:
                    st.error(f"Error performing calculation: {str(e)}")
        else:
            st.warning("Please enter at least one word to perform calculation.")

if __name__ == "__main__":
    main()
