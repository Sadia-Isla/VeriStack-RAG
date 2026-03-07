import sys
import os
# Essential: Add local directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import asyncio
from services import RAGEngine

st.set_page_config(page_title="VeriStack RAG", page_icon="🛡️")

# Sidebar for credentials
with st.sidebar:
    st.header("🔑 Configuration")
    user_openai_key = st.text_input("OpenAI API Key", type="password")
    qdrant_url = st.text_input("Qdrant URL", placeholder="https://xxx.cloud.qdrant.io")
    qdrant_api_key = st.text_input("Qdrant API Key", type="password")
    
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    index_button = st.button("Index Document")

# Engine initialization
@st.cache_resource(show_spinner=False)
def get_engine(openai_key, q_url, q_key):
    if not (openai_key and q_url and q_key):
        return None
    os.environ["OPENAI_API_KEY"] = openai_key
    os.environ["QDRANT_URL"] = q_url
    os.environ["QDRANT_API_KEY"] = q_key
    return RAGEngine()

engine = get_engine(user_openai_key, qdrant_url, qdrant_api_key)

if not engine:
    st.warning("Please provide all keys in the sidebar.")
    st.stop()

# Indexing Logic
if uploaded_file and index_button:
    with st.spinner("Indexing..."):
        if not os.path.exists("temp"): os.makedirs("temp")
        path = os.path.join("temp", uploaded_file.name)
        with open(path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        asyncio.run(engine.process_pdf(path))
        st.success("Indexed!")

# Chat Logic
if prompt := st.chat_input("Ask a question:"):
    with st.chat_message("user"): st.write(prompt)
    with st.chat_message("assistant"):
        res = asyncio.run(engine.query(prompt, 3))
        st.write(res["answer"])
