import sys
import os
import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from services import RAGEngine

st.set_page_config(page_title="VeriStack RAG", page_icon="🛡️")

# --- UI Styling ---
st.markdown("""<style> .stChatMessage { border-radius: 15px; margin-bottom: 10px; } </style>""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("🛡️ VeriStack Admin")
    st.info("**Step 1:** Enter Keys\n**Step 2:** Upload PDF\n**Step 3:** Index & Chat")
    
    user_key = st.text_input("OpenAI Key", type="password")
    q_url = st.text_input("Qdrant URL")
    q_key = st.text_input("Qdrant Key", type="password")
    
    st.divider()
    uploaded_file = st.file_uploader("📂 Knowledge Base", type="pdf")
    col1, col2 = st.columns(2)
    with col1:
        index_btn = st.button("🚀 Index", type="primary")
    with col2:
        # NEW: Clear button to wipe old "junk" data from Qdrant
        clear_btn = st.button("🗑️ Reset")

@st.cache_resource
def get_engine(key, url, qk):
    if not (key and url and qk): return None
    os.environ["OPENAI_API_KEY"] = key
    os.environ["QDRANT_URL"] = url
    os.environ["QDRANT_API_KEY"] = qk
    return RAGEngine()

engine = get_engine(user_key, q_url, q_key)

if clear_btn and engine:
    engine.client.delete_collection("docs")
    st.sidebar.success("Collection Wiped!")
    st.rerun()

st.title("🛡️ AI Document Assistant")

if not engine:
    st.warning("Please enter your credentials in the sidebar.")
    st.stop()

if uploaded_file and index_btn:
    with st.spinner("Cleaning & Indexing Document..."):
        path = f"temp_{uploaded_file.name}"
        with open(path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        engine.process_pdf(path)
        st.success("Indexing Complete!")

for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if prompt := st.chat_input("Ask me anything about the document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Scanning..."):
            res = engine.query(prompt, 5)
            st.markdown(res["answer"])
            if res["sources"]:
                with st.expander("References"):
                    for s in res["sources"]:
                        st.caption(f"Score: {s['score']:.2f}")
                        st.write(s['text'])
            st.session_state.messages.append({"role": "assistant", "content": res["answer"]})
