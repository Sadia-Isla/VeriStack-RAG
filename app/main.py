import streamlit as st
import os
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
    
    ready = user_openai_key and qdrant_url and qdrant_api_key
    index_button = st.button("Index", disabled=not ready)

# --- INITIALIZE ENGINE ---
@st.cache_resource(show_spinner=False)
def get_engine(openai_key, q_url, q_key):
    if not (openai_key and q_url and q_key):
        return None
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
        if not os.path.exists("temp_data"): os.makedirs("temp_data")
        temp_path = f"temp_data/{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        asyncio.run(engine.process_pdf(temp_path))
        st.success(f"Successfully Indexed: {uploaded_file.name}")

# Chat session
if "messages" not in st.session_state: st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.write(msg["content"])

if prompt := st.chat_input("Ask about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Querying..."):
            res = asyncio.run(engine.query(prompt, top_k=3))
            st.write(res["answer"])
            if res["sources"]:
                with st.expander("View Sources"):
                    for src in res["sources"]:
                        st.caption(f"Score: {src['score']:.4f}")
                        st.write(src['text'])
            st.session_state.messages.append({"role": "assistant", "content": res["answer"]})
