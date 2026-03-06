
import streamlit as st
import requests

st.title("🛡️ VeriStack RAG")

# Sidebar Ingestion
with st.sidebar:
    st.header("Admin: Upload Data")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    if uploaded_file and st.button("Index"):
        files = {"file": uploaded_file}
        res = requests.post("http://localhost:8000/ingest", files=files)
        st.success("Indexed!")

# Chat Interface
if "messages" not in st.session_state: st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.write(prompt)

    with st.chat_message("assistant"):
        response = requests.post("http://localhost:8000/query", json={"query": prompt})
        data = response.json()
        st.write(data["answer"])
        
        with st.expander("View Citations"):
            for src in data["sources"]:
                st.info(f"Source (Score: {src['score']:.2f}):\n{src['text'][:200]}...")
    
    st.session_state.messages.append({"role": "assistant", "content": data["answer"]})
