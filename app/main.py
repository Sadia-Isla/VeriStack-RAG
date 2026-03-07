import sys
import os
import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from services import RAGEngine

# Page Styling
st.set_page_config(page_title="VeriStack RAG", page_icon="🛡️", layout="centered")

# Custom CSS for a modern look
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 20px; border: 1px solid #ff4b4b; }
    .stTextInput>div>div>input { border-radius: 10px; }
    .chat-bubble { padding: 15px; border-radius: 15px; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- SIDEBAR: CONFIG & GUIDE ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com", width=80)
    st.title("🛡️ Control Center")
    
    with st.expander("📖 How to use this App", expanded=True):
        st.info("""
        1. **Connect**: Enter your OpenAI and Qdrant Keys below.
        2. **Upload**: Drop a PDF in the 'Knowledge Base' section.
        3. **Index**: Click 'Start Indexing' to process the text.
        4. **Chat**: Use the chat box to ask questions or request summaries!
        """)

    st.subheader("🔑 API Credentials")
    user_key = st.text_input("OpenAI API Key", type="password")
    q_url = st.text_input("Qdrant Endpoint", placeholder="https://...")
    q_key = st.text_input("Qdrant API Key", type="password")
    
    st.divider()
    st.subheader("📂 Knowledge Base")
    uploaded_file = st.file_uploader("Choose a PDF document", type="pdf")
    index_btn = st.button("🚀 Start Indexing", type="primary")

# --- ENGINE LOGIC ---
@st.cache_resource
def get_engine(key, url, qk):
    if not (key and url and qk): return None
    os.environ["OPENAI_API_KEY"] = key
    os.environ["QDRANT_URL"] = url
    os.environ["QDRANT_API_KEY"] = qk
    return RAGEngine()

engine = get_engine(user_key, q_url, q_key)

# --- MAIN PAGE UI ---
st.title("🛡️ VeriStack AI Assistant")
st.caption("Intelligent Document Analysis & RAG Platform")

if not engine:
    st.warning("⚠️ Access Denied: Please provide API keys in the sidebar to unlock the assistant.")
    st.stop()

# Indexing Execution
if uploaded_file and index_btn:
    with st.status("🔍 Analyzing Document...", expanded=True) as status:
        st.write("Extracting text...")
        path = f"temp_{uploaded_file.name}"
        with open(path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.write("Generating Vector Embeddings...")
        engine.process_pdf(path)
        status.update(label="✅ Knowledge Base Updated!", state="complete", expanded=False)
    st.balloons()

# Chat Interface
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("What would you like to know?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            res = engine.query(prompt, 5) # Increased top_k for better summaries
            st.markdown(res["answer"])
            
            if res["sources"]:
                with st.expander("📍 Verified Sources"):
                    for src in res["sources"]:
                        st.caption(f"Relevance Score: {src['score']:.4f}")
                        st.write(src['text'])
                        st.divider()
            
            st.session_state.messages.append({"role": "assistant", "content": res["answer"]})
