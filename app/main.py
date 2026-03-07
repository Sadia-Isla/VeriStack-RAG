import sys
import os
import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from services import RAGEngine

# --- PAGE SETUP ---
st.set_page_config(page_title="VeriStack RAG", page_icon="🛡️", layout="wide")

# Custom UI Styling
st.markdown("""
    <style>
    .stChatMessage { border-radius: 20px; background-color: #f0f2f6; border: 1px solid #ddd; }
    .stButton>button { border-radius: 10px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- SIDEBAR: CONFIG & LINKS ---
with st.sidebar:
    # Replace st.image with this:
    with st.sidebar:
    # This centers the icon and adds a bit of spacing
    _, col_img, _ = st.columns([1, 2, 1])
    with col_img:
        st.image("https://img.icons8.com")
    
    st.markdown("<h2 style='text-align: center;'>VeriStack RAG</h2>", unsafe_allow_html=True)
    st.divider())
    st.title("🛡️ VeriStack RAG")
    
    with st.expander("🔑 Need API Keys?", expanded=False):
        st.markdown("""
        **1. OpenAI Key**  
        Create one at: [platform.openai.com](https://platform.openai.com)
        
        **2. Qdrant Cloud**  
        Get your Cluster URL and Key at: [cloud.qdrant.io](https://cloud.qdrant.io)
        """)

    st.subheader("⚙️ Setup Credentials")
    u_key = st.text_input("OpenAI API Key", type="password")
    q_url = st.text_input("Qdrant Endpoint", placeholder="https://xxx.cloud.qdrant.io")
    q_key = st.text_input("Qdrant API Key", type="password")
    
    st.divider()
    st.subheader("📂 Knowledge Base")
    up_file = st.file_uploader("Upload your PDF Document", type="pdf")
    idx_btn = st.button("🚀 Start Indexing", use_container_width=True, type="primary")

# --- ENGINE INITIALIZATION ---
@st.cache_resource
def get_engine(k, u, qk):
    if not (k and u and qk): return None
    os.environ["OPENAI_API_KEY"], os.environ["QDRANT_URL"], os.environ["QDRANT_API_KEY"] = k, u, qk
    return RAGEngine()

engine = get_engine(u_key, q_url, q_key)

# --- MAIN INTERFACE ---
st.title("🛡️ VeriStack RAG")
st.caption("Secure AI Intelligence Layer for Document Analysis")

if not engine:
    st.info("👋 Welcome! To begin, please enter your **API Keys** in the sidebar. Once connected, you can upload and query any PDF.")
    st.stop()

# Indexing Logic
if up_file and idx_btn:
    with st.status("Reading document content...", expanded=True) as s:
        path = f"temp_{up_file.name}"
        with open(path, "wb") as f: f.write(up_file.getbuffer())
        engine.process_pdf(path)
        s.update(label="✅ Knowledge Base Synchronized!", state="complete")
    st.balloons()

# Chat Display
for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

# Query Execution
if prompt := st.chat_input("Ask a question about the document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            res = engine.query(prompt, 5)
            st.markdown(res["answer"])
            
            if res["sources"]:
                with st.expander("📍 Verified Document Sources"):
                    for s in res["sources"]:
                        st.caption(f"Relevance Score: {s['score']:.2f}")
                        st.write(s['text'])
                        st.divider()
            
            st.session_state.messages.append({"role": "assistant", "content": res["answer"]})
