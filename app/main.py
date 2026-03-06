import streamlit as st
import os
import shutil
import asyncio
# 1. Import your existing backend logic
from app.services import RAGEngine
from app.models import QueryRequest

st.set_page_config(page_title="VeriStack RAG", page_icon="🛡️")
st.title("🛡️ VeriStack RAG")

# 2. Initialize the Engine (Cache it so it doesn't reload every click)
@st.cache_resource
def get_engine():
    if not os.path.exists("data"):
        os.makedirs("data")
    return RAGEngine()

engine = get_engine()

# Sidebar Ingestion
with st.sidebar:
    st.header("Admin: Upload Data")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    
    if uploaded_file and st.button("Index"):
        with st.spinner("Processing PDF..."):
            # Replicating your FastAPI /ingest logic
            temp_path = f"data/{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Since your engine uses 'async def', we run it with asyncio
            asyncio.run(engine.process_pdf(temp_path))
            st.success(f"Indexed: {uploaded_file.name}")

# Chat Interface
if "messages" not in st.session_state: 
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): 
        st.write(msg["content"])

if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): 
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Replicating your FastAPI /query logic
            # Using asyncio.run because your engine.query is async
            data = asyncio.run(engine.query(prompt, top_k=3))
            
            # 'data' is now your QueryResponse object
            st.write(data.answer)
            
            if hasattr(data, 'sources') and data.sources:
                with st.expander("View Citations"):
                    for src in data.sources:
                        # Adjusted to object notation (src.score) since it's a model
                        st.info(f"Source (Score: {src.score:.2f}):\n{src.text[:200]}...")
    
    st.session_state.messages.append({"role": "assistant", "content": data.answer})
