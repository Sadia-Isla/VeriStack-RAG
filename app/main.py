import streamlit as st
import os
import asyncio
from services import RAGEngine

st.set_page_config(page_title="VeriStack RAG", page_icon="🛡️", layout="wide")

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("🔑 API Configuration")
    user_openai_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    
    st.divider()
    st.header("☁️ Qdrant Cloud")
    qdrant_url = st.text_input("Endpoint URL", placeholder="https://xxx.cloud.qdrant.io")
    qdrant_api_key = st.text_input("Qdrant API Key", type="password")
    
    st.divider()
    st.header("📂 Data Management")
    uploaded_file = st.file_uploader("Upload PDF Knowledge", type="pdf")
    
    # Validation for buttons
    keys_ready = user_openai_key and qdrant_url and qdrant_api_key
    if st.button("Index Document", disabled=not (uploaded_file and keys_ready)):
        with st.spinner("Indexing..."):
            # Update environment for the engine
            os.environ["OPENAI_API_KEY"] = user_openai_key
            os.environ["QDRANT_URL"] = qdrant_url
            os.environ["QDRANT_API_KEY"] = qdrant_api_key
            
            # Temporary save for SimpleDirectoryReader
            if not os.path.exists("temp_data"): os.makedirs("temp_data")
            temp_path = os.path.join("temp_data", uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            engine = RAGEngine()
            asyncio.run(engine.process_pdf(temp_path))
            st.success(f"Successfully indexed {uploaded_file.name}")

# --- Main Chat Interface ---
st.title("🛡️ VeriStack RAG")

if not keys_ready:
    st.info("👈 Please enter your API keys in the sidebar to start.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        # Setup environment again to ensure engine has keys
        os.environ["OPENAI_API_KEY"] = user_openai_key
        os.environ["QDRANT_URL"] = qdrant_url
        os.environ["QDRANT_API_KEY"] = qdrant_api_key
        
        engine = RAGEngine()
        with st.spinner("Searching..."):
            res = asyncio.run(engine.query(prompt, top_k=3))
            st.write(res["answer"])
            
            with st.expander("View Citations"):
                for src in res["sources"]:
                    st.caption(f"Score: {src['score']:.2f}")
                    st.write(f"{src['text'][:300]}...")

    st.session_state.messages.append({"role": "assistant", "content": res["answer"]})
