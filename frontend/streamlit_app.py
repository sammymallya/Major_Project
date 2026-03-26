"""
Professional Streamlit Frontend for QA Pipeline

A dark-themed ChatGPT-like interface for querying the pipeline with
support for multiple retrieval modes: none, vectordb, kg, hybrid.
"""

import streamlit as st
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from backend.services.orchestration import run_pipeline

# Configure Streamlit theme and layout
st.set_page_config(
    page_title="QA Pipeline",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Apply custom dark theme styling
st.markdown("""
<style>
    :root {
        --primary-color: #1f1f1f;
        --secondary-color: #2d2d2d;
        --tertiary-color: #3d3d3d;
    }
    
    /* Main background */
    .stApp {
        background-color: #0d0d0d;
        color: #ececec;
    }
    
    /* Chat container */
    .chat-container {
        height: 100%;
        display: flex;
        flex-direction: column;
    }
    
    /* User message */
    .user-message {
        background-color: #10a37f;
        color: white;
        border-radius: 12px;
        padding: 12px 16px;
        margin: 8px 0;
        margin-left: auto;
        max-width: 70%;
        word-wrap: break-word;
    }
    
    /* Assistant message */
    .assistant-message {
        background-color: #2d2d2d;
        color: #ececec;
        border-radius: 12px;
        padding: 12px 16px;
        margin: 8px 0;
        margin-right: auto;
        max-width: 85%;
        word-wrap: break-word;
    }
    
    /* Input area */
    .input-container {
        border-top: 1px solid #3d3d3d;
        padding: 20px;
        background-color: #0d0d0d;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #10a37f;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 10px 20px;
        font-weight: 600;
        cursor: pointer;
        transition: background-color 0.2s;
    }
    
    .stButton > button:hover {
        background-color: #1a7f5d;
    }
    
    /* Select box */
    .stSelectbox [role="listbox"] {
        background-color: #2d2d2d !important;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state for chat history."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "mode" not in st.session_state:
        st.session_state.mode = "hybrid"


def add_to_history(query: str, answer: str, mode: str):
    """Add a message pair to chat history."""
    st.session_state.chat_history.append({
        "type": "user",
        "content": query,
    })
    st.session_state.chat_history.append({
        "type": "assistant",
        "content": answer,
        "mode": mode,
    })


def display_chat_history():
    """Display the chat history in a ChatGPT-like format."""
    for message in st.session_state.chat_history:
        if message["type"] == "user":
            st.markdown(f"""
            <div style='display: flex; justify-content: flex-end; margin: 12px 0;'>
                <div style='background-color: #10a37f; color: white; border-radius: 12px; padding: 12px 16px; max-width: 70%; word-wrap: break-word;'>
                    {message['content']}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            mode = message.get("mode", "unknown")
            st.markdown(f"""
            <div style='display: flex; justify-content: flex-start; margin: 12px 0;'>
                <div style='background-color: #2d2d2d; color: #ececec; border-radius: 12px; padding: 12px 16px; max-width: 85%; word-wrap: break-word;'>
                    <div style='font-size: 12px; color: #888; margin-bottom: 8px;'>Mode: <strong>{mode}</strong></div>
                    {message['content']}
                </div>
            </div>
            """, unsafe_allow_html=True)


def main():
    """Main Streamlit app."""
    initialize_session_state()
    
    # Header
    st.markdown("<h1 style='text-align: center; color: #10a37f; margin-bottom: 30px;'>💬 QA Pipeline</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #888; margin-bottom: 30px;'>Query your knowledge base with multiple retrieval modes</p>", unsafe_allow_html=True)
    
    # Chat display area
    st.markdown("<div class='chat-container' style='margin-bottom: 50px;'>", unsafe_allow_html=True)
    
    if st.session_state.chat_history:
        display_chat_history()
    else:
        st.markdown(
            "<div style='text-align: center; color: #666; margin-top: 60px;'>"
            "<h3>👋 Welcome to the QA Pipeline</h3>"
            "<p>Start by typing a question below and selecting your retrieval mode.</p>"
            "</div>",
            unsafe_allow_html=True
        )
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Input area with fixed positioning
    st.markdown("<hr style='border-color: #3d3d3d;'>", unsafe_allow_html=True)
    
    with st.container():
        col1, col2, col3 = st.columns([3, 1, 0.5])
        
        with col1:
            user_query = st.text_input(
                label="Your question",
                placeholder="Ask a question about tourism in Karnataka...",
                label_visibility="collapsed",
                key="user_query_input"
            )
        
        with col2:
            mode = st.selectbox(
                label="Mode",
                options=["none", "vectordb", "kg", "hybrid"],
                index=3,  # Default to hybrid
                key="mode_select",
                label_visibility="collapsed"
            )
        
        with col3:
            send_button = st.button("Send", use_container_width=True, key="send_btn")
    
    # Process query when send button is clicked
    if send_button and user_query.strip():
        with st.spinner("🔍 Processing your query..."):
            try:
                # Run the pipeline
                response = run_pipeline(query=user_query, test_mode=mode)
                answer = response.answer
                
                # Add to history
                add_to_history(user_query, answer, mode)
                
                # Rerun to display updated history and clear input
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error processing query: {str(e)}")
                st.error("Please try again with a different query or check the pipeline configuration.")


if __name__ == "__main__":
    main()
