import sys
import os
# Force the current 'app' directory into the path so 'services' can be found
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import shutil
import asyncio
from services import RAGEngine

st.set_page_config(page_title="VeriStack RAG", page_icon="🛡️")

# --- SIDEBAR ---
with st.sidebar:
    st.header("🔑 API Configuration")
    user_openai_key = st.text_input("OpenAI API Key", type="password")
    
    st.divider()
    st.header("Storage: Qdrant Cloud")
    qdrant_url = st.text_input("Qdrant Endpoint URL", placeholder="https://xxx.cloud.qdrant.io")
    qdrant_api_key = st.text_input("Qdrant API Key", type="password")
    
    st.divider()
    st.header("Admin: Upload Data")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    
    # Check if all required keys are provided
    ready = user_openai_key and qdrant_url and qdrant_api_key
    index_button = st.button("Index", disabled=not ready)

# --- INITIALIZE ENGINE ---
@st.cache_resource(show_spinner=False)
def get_engine(openai_key, q_url, q_key):
    if not (openai_key and q_url and q_key):
        return None
    # Set environment variables for the RAGEngine to pick up
    os.environ["OPENAI_API_KEY"] = openai_key
    os.environ["QDRANT_URL"] = q_url
    os.environ["QDRANT_API_KEY"] = q_key
    return RAGEngine()

engine = get_engine(user_openai_key, qdrant_url, qdrant_api_key)

# --- UI LOGIC ---
st.title("🛡️ VeriStack RAG")

if not engine:
    st.info("👈 Please enter your API keys in the sidebar to begin.")
    st.stop()

if uploaded_file and index_button:
    with st.spinner("Indexing PDF to Qdrant Cloud..."):
        if not os.path.exists("temp_data"): 
            os.makedirs("temp_data")
        temp_path = f"temp_data/{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Run async processing
        asyncio.run(engine.process_pdf(temp_path))
        st.success(f"Successfully Indexed: {uploaded_file.name}")

# --- CHAT SESSION ---
if "messages" not in st.session_state: 
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): 
        st.write(msg["content"])

# Handle new user input
if prompt := st.chat_input("Ask about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): 
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Querying..."):
            # Execute async query
            res = asyncio.run(engine.query(prompt, top_k=3))
            st.write(res["answer"])
            
            # Show citation sources if available
            if res["sources"]:
                with st.expander("View Sources"):
                    for src in res["sources"]:
                        st.caption(f"Score: {src['score']:.4f}")
                        st.write(src['text'])
            
            st.session_state.messages.append({"role": "assistant", "content": res["answer"]})
