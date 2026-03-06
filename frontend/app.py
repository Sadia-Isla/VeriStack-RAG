import streamlit as st
# 1. IMPORT YOUR BACKEND LOGIC DIRECTLY
# Replace 'backend_logic' with the actual filename where your RAG functions live
from main import process_ingest_logic, process_query_logic 

st.set_page_config(page_title="VeriStack RAG", page_icon="🛡️")
st.title("🛡️ VeriStack RAG")

# Sidebar Ingestion
with st.sidebar:
    st.header("Admin: Upload Data")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    if uploaded_file and st.button("Index"):
        with st.spinner("Indexing..."):
            # 2. CALL THE FUNCTION DIRECTLY (No requests.post)
            # You might need to pass uploaded_file.getvalue() or the file object
            success = process_ingest_logic(uploaded_file)
            if success:
                st.success("Indexed!")

# Chat Interface
if "messages" not in st.session_state: 
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): 
        st.write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): 
        st.write(prompt)

    with st.chat_message("assistant"):
        # 3. CALL THE QUERY FUNCTION DIRECTLY
        data = process_query_logic(prompt) 
        st.write(data["answer"])
        
        with st.expander("View Citations"):
            for src in data.get("sources", []):
                st.info(f"Source (Score: {src['score']:.2f}):\n{src['text'][:200]}...")
    
    st.session_state.messages.append({"role": "assistant", "content": data["answer"]})
