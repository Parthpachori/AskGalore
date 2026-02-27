import sys
from unittest.mock import MagicMock
# Mock transformers to prevent it from loading broken torch DLLs
sys.modules["transformers"] = MagicMock()

import streamlit as st
import os
from dotenv import load_dotenv
from src.graph import app_graph
from src.rag_engine import RAGEngine

load_dotenv()

# Page Config
st.set_page_config(
    page_title="AskGalore - RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stChatFloatingInputContainer {
        bottom: 20px;
    }
    .stChatMessage {
        background-color: #1e1e1e;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
        border: 1px solid #333;
    }
    .source-box {
        background-color: #262730;
        padding: 10px;
        border-radius: 5px;
        font-size: 0.8em;
        margin-top: 10px;
        border-left: 3px solid #00ffa2;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = RAGEngine()

# Sidebar for Knowledge Base Management
with st.sidebar:
    st.title("Settings")
    st.write("Manage your knowledge base.")
    
    uploaded_files = st.file_uploader("Upload Documents (PDF/TXT)", accept_multiple_files=True, type=["pdf", "txt"])
    
    if st.button("Index Documents"):
        if uploaded_files:
            if not os.path.exists("data"):
                os.makedirs("data")
            
            for uploaded_file in uploaded_files:
                with open(os.path.join("data", uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())
            
            with st.spinner("Ingesting documents..."):
                st.session_state.rag_engine.ingest_documents()
                st.success("Indexing complete!")
        else:
            st.warning("Please upload files first.")

    st.divider()
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Main Chat Interface
st.title("🤖 AskGalore AI Avatar")
st.caption("RAG-powered intelligence with LangGraph orchestration")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.markdown(f'<div class="source-box">{source}</div>', unsafe_allow_html=True)

# Chat Input
if prompt := st.chat_input("Ask me anything about your documents..."):
    # Add user message to state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process via LangGraph
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Prepare initial state
                initial_state = {
                    "query": prompt,
                    "chat_history": st.session_state.messages[:-1],
                    "retrieved_docs": [],
                    "source_documents": [],
                    "generation": "",
                    "intent": None,
                    "context_valid": False,
                    "confidence_score": 0.0,
                    "error": None
                }
                
                # Run the graph
                final_output = app_graph.invoke(initial_state)
                
                response = final_output.get("generation", "I'm sorry, something went wrong.")
                sources = final_output.get("source_documents", [])
                
                st.markdown(response)
                
                if sources:
                    with st.expander("Sources"):
                        for source in sources:
                            st.markdown(f'<div class="source-box">{source}</div>', unsafe_allow_html=True)
                
                # Add to history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response,
                    "sources": sources
                })
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.session_state.messages.append({"role": "assistant", "content": f"Error: {str(e)}"})
