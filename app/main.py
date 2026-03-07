import sys
import os
import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from services import RAGEngine

st.set_page_config(page_title="VeriStack RAG", page_icon="🛡️")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.title("🛡️ Config")
    user_key = st.text_input("OpenAI Key", type="password")
    q_url = st.text_input("Qdrant URL")
    q_key = st.text_input("Qdrant Key", type="password")
    
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    index_btn = st.button("Index")

@st.cache_resource
def get_engine(key, url, qk):
    if not (key and url and qk): return None
    os.environ["OPENAI_API_KEY"] = key
    os.environ["QDRANT_URL"] = url
    os.environ["QDRANT_API_KEY"] = qk
    return RAGEngine()

engine = get_engine(user_key, q_url, q_key)
st.title("VeriStack Assistant")

if not engine:
    st.info("Fill sidebar to start.")
    st.stop()

# Indexing (Sync)
if uploaded_file and index_btn:
    with st.spinner("Indexing..."):
        path = f"temp_{uploaded_file.name}"
        with open(path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        engine.process_pdf(path)
        st.success("Ready!")

# Chat Display
for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

# Querying (Sync)
if prompt := st.chat_input("Ask away..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        res = engine.query(prompt, 3)
        st.markdown(res["answer"])
        st.session_state.messages.append({"role": "assistant", "content": res["answer"]})
