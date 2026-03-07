import sys
import os
import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from services import RAGEngine

st.set_page_config(page_title="VeriStack AI", page_icon="🛡️", layout="wide")

# Custom UI Styling
st.markdown("""
    <style>
    .stChatMessage { border-radius: 20px; background-color: #f0f2f6; border: 1px solid #ddd; }
    .stButton>button { border-radius: 10px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar Instruction Guide
with st.sidebar:
    st.title("🛡️ Setup Guide")
    st.markdown("""
    ### **How to start:**
    1. 🔑 **Keys**: Enter OpenAI & Qdrant details.
    2. 📂 **Upload**: Select your PDF.
    3. 🚀 **Index**: Click Index to 'teach' the AI.
    4. 💬 **Chat**: Ask for a summary or details!
    """)
    
    u_key = st.text_input("OpenAI Key", type="password")
    q_url = st.text_input("Qdrant URL")
    q_key = st.text_input("Qdrant Key", type="password")
    
    st.divider()
    up_file = st.file_uploader("Upload Policy PDF", type="pdf")
    idx_btn = st.button("🚀 Start Indexing", use_container_width=True)

@st.cache_resource
def get_engine(k, u, qk):
    if not (k and u and qk): return None
    os.environ["OPENAI_API_KEY"], os.environ["QDRANT_URL"], os.environ["QDRANT_API_KEY"] = k, u, qk
    return RAGEngine()

engine = get_engine(u_key, q_url, q_key)

st.title("🛡️ VeriStack Intelligence Portal")

if not engine:
    st.info("👋 Welcome! Please enter your API credentials in the sidebar to begin.")
    st.stop()

if up_file and idx_btn:
    with st.status("Reading document content...", expanded=True) as s:
        path = f"temp_{up_file.name}"
        with open(path, "wb") as f: f.write(up_file.getbuffer())
        engine.process_pdf(path)
        s.update(label="✅ Knowledge Base Ready!", state="complete")

# Chat
for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if prompt := st.chat_input("Ask about the document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            res = engine.query(prompt, 5)
            st.markdown(res["answer"])
            if res["sources"]:
                with st.expander("📍 View Document References"):
                    for s in res["sources"]:
                        st.caption(f"Relevance: {s['score']:.2f}")
                        st.write(s['text'])
            st.session_state.messages.append({"role": "assistant", "content": res["answer"]})
