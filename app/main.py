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
    
    # Enable button only if all 3 keys are present
    ready = user_openai_key and qdrant_url and qdrant_api_key
    index_button = st.button("Index", disabled=not ready)

@st.cache_resource(show_spinner=False)
def get_engine(openai_key, q_url, q_key):
    if not (openai_key and q_url and q_key):
        return None
    
    # Set environment variables for the engine to pick up
    os.environ["OPENAI_API_KEY"] = openai_key
    os.environ["QDRANT_URL"] = q_url
    os.environ["QDRANT_API_KEY"] = q_key
    
    return RAGEngine()

engine = get_engine(user_openai_key, qdrant_url, qdrant_api_key)
