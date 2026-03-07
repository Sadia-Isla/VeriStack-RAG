import sys
import os
import asyncio
import streamlit as st

# Force add the current directory to path for local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from services import RAGEngine

st.set_page_config(page_title="VeriStack RAG", page_icon="🛡️", layout="wide")

# --- ASYNC HELPER ---
def run_async(coro):
    """Creates a fresh event loop for every Streamlit rerun to avoid closure errors."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()

# --- INITIALIZE SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- SIDEBAR ---
with st.sidebar:
    st.title("🛡️ Configuration")
    user_openai_key = st.text_input("OpenAI API Key", type="password")
    qdrant_url = st.text_input("Qdrant URL", placeholder="https://xxx.cloud.qdrant.io")
    qdrant_api_key = st.text_input("Qdrant API Key", type="password")
    
    st.divider()
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    index_button = st.button("Index Document")

# --- ENGINE SETUP ---
@st.cache_resource(show_spinner=False)
def get_engine(openai_key, q_url, q_key):
    if not (openai_key and q_url and q_key):
        return None
    os.environ["OPENAI_API_KEY"] = openai_key
    os.environ["QDRANT_URL"] = q_url
    os.environ["QDRANT_API_KEY"] = q_key
    return RAGEngine()

engine = get_engine(user_openai_key, qdrant_url, qdrant_api_key)

# --- MAIN UI ---
st.title("VeriStack RAG Assistant")

if not engine:
    st.info("👈 Please enter your API keys and Qdrant details in the sidebar.")
    st.stop()

# Handle PDF Indexing
if uploaded_file and index_button:
    with st.spinner("Indexing to Qdrant Cloud..."):
        os.makedirs("temp", exist_ok=True)
        path = os.path.join("temp", uploaded_file.name)
        with open(path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        run_async(engine.process_pdf(path))
        st.success(f"Indexed: {uploaded_file.name}")

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat Input
if prompt := st.chat_input("Ask a question about your documents..."):
    # 1. User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Assistant Message
    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            res = run_async(engine.query(prompt, top_k=3))
            full_response = res["answer"]
            
            st.markdown(full_response)
            
            # Show sources if available
            if res["sources"]:
                with st.expander("View Sources"):
                    for src in res["sources"]:
                        st.caption(f"Score: {src['score']:.4f}")
                        st.write(src['text'])
            
            # Save to history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
