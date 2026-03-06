import streamlit as st
import os
import shutil
import asyncio

# Fix: Use direct imports since main.py is inside the /app folder
from services import RAGEngine
from models import QueryRequest

st.set_page_config(page_title="VeriStack RAG", page_icon="🛡️")

# --- SIDEBAR: API KEY & UPLOAD ---
with st.sidebar:
    st.header("🔑 Authentication")
    user_api_key = st.text_input(
        "Enter your OpenAI API Key", 
        type="password", 
        placeholder="sk-...",
        help="This key is used only for this session and is not stored."
    )
    
    st.divider()
    
    st.header("Admin: Upload Data")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    
    # Only show the Index button if a key is provided
    index_button = st.button("Index", disabled=not user_api_key)
    
    if not user_api_key:
        st.warning("Please enter an API Key to enable indexing.")

# --- ENGINE INITIALIZATION ---
# We cache the engine based on the API key. 
# If the key changes, a new engine is created.
@st.cache_resource(show_spinner=False)
def get_engine(api_key):
    if not api_key:
        return None
    # Set the environment variable so LlamaIndex/OpenAI can find it
    os.environ["OPENAI_API_KEY"] = api_key
    
    if not os.path.exists("data"):
        os.makedirs("data")
    return RAGEngine()

engine = get_engine(user_api_key)

# --- APP LOGIC ---
st.title("🛡️ VeriStack RAG")

if not user_api_key:
    st.info("👈 Please enter your OpenAI API Key in the sidebar to start chatting.")
    st.stop() # Stops the app from running further until key is entered

# Handle Indexing
if uploaded_file and index_button:
    with st.spinner("Processing PDF..."):
        temp_path = f"data/{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
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
        with st.spinner("Searching and generating..."):
            try:
                data = asyncio.run(engine.query(prompt, top_k=3))
                st.write(data.answer)
                
                if hasattr(data, 'sources') and data.sources:
                    with st.expander("View Citations"):
                        for src in data.sources:
                            st.info(f"Source (Score: {src.score:.2f}):\n{src.text[:200]}...")
                
                st.session_state.messages.append({"role": "assistant", "content": data.answer})
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Check if your API key is valid and has enough credits.")
