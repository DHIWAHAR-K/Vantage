"""
Streamlit demo for text-to-SQL model.

Interactive web interface to test the fine-tuned T5 model.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from src.inference import Text2SQLInference


# Page config
st.set_page_config(
    page_title="Vantage Text-to-SQL",
    page_icon="🔍",
    layout="wide"
)

# Title
st.title("🔍 Vantage Text-to-SQL")
st.markdown("**Convert natural language questions to SQL queries**")

# Sidebar
with st.sidebar:
    st.header("Settings")
    
    checkpoint = st.text_input(
        "Model Checkpoint",
        value="checkpoints/final",
        help="Path to fine-tuned model"
    )
    
    max_length = st.slider(
        "Max SQL Length",
        min_value=32,
        max_value=512,
        value=128,
        help="Maximum tokens in generated SQL"
    )
    
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        help="0 = greedy (deterministic), higher = more random"
    )

# Load model (cached)
@st.cache_resource
def load_model(checkpoint_path: str, max_len: int, temp: float):
    """Load model (cached)."""
    try:
        return Text2SQLInference(
            checkpoint_path=checkpoint_path,
            max_length=max_len,
            temperature=temp
        )
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("Input")
    
    # Question input
    question = st.text_area(
        "Question",
        placeholder="How many users signed up today?",
        height=100,
        help="Natural language question"
    )
    
    # Schema input
    schema = st.text_area(
        "Database Schema",
        placeholder="users(id, name, email, signup_date)",
        height=150,
        help="Table definitions (e.g., 'users(id, name) | posts(id, user_id, title)')"
    )
    
    # Generate button
    generate_btn = st.button("🚀 Generate SQL", type="primary", use_container_width=True)

with col2:
    st.subheader("Generated SQL")
    
    if generate_btn:
        if not question or not schema:
            st.warning("Please provide both question and schema")
        else:
            # Load model
            model = load_model(checkpoint, max_length, temperature)
            
            if model is None:
                st.error("Model not loaded. Check checkpoint path in sidebar.")
            else:
                # Generate
                with st.spinner("Generating SQL..."):
                    try:
                        sql = model.generate(question, schema)
                        
                        # Display result
                        st.code(sql, language="sql")
                        
                        # Copy button
                        st.button("📋 Copy to Clipboard")
                        
                        # Execution hint
                        st.info("💡 **Tip**: Always validate generated SQL before running on production data")
                        
                    except Exception as e:
                        st.error(f"Generation failed: {e}")
    else:
        st.info("Enter a question and schema, then click 'Generate SQL'")

# Examples
with st.expander("📚 Example Queries"):
    st.markdown("""
    ### Example 1: Simple Count
    **Question**: How many users are there?  
    **Schema**: `users(id, name, email)`  
    **Expected SQL**: `SELECT COUNT(*) FROM users`
    
    ### Example 2: Filtering
    **Question**: Show all active users  
    **Schema**: `users(id, name, email, active)`  
    **Expected SQL**: `SELECT * FROM users WHERE active = 1`
    
    ### Example 3: Join
    **Question**: List all users and their posts  
    **Schema**: `users(id, name) | posts(id, user_id, title)`  
    **Expected SQL**: `SELECT users.name, posts.title FROM users JOIN posts ON users.id = posts.user_id`
    
    ### Example 4: Aggregation
    **Question**: Count posts per user  
    **Schema**: `users(id, name) | posts(id, user_id, title)`  
    **Expected SQL**: `SELECT users.name, COUNT(posts.id) FROM users JOIN posts ON users.id = posts.user_id GROUP BY users.id`
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Powered by T5-small fine-tuned on SynSQL | Built with MLX on Apple Silicon</p>
</div>
""", unsafe_allow_html=True)
