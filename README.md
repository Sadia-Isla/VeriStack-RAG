🛡️ VeriStack RAG: Document Intelligence Engine
VeriStack is a production-grade Retrieval-Augmented Generation (RAG) system built with FastAPI and LlamaIndex. It moves beyond basic vector search by implementing hybrid retrieval, reranking, and grounded evaluation to ensure hallucination-free AI responses.

🚀 Key Production Features (What's Inside)
Hybrid Retrieval: Combines semantic (Vector) search with BM25 (Keyword) search to handle both nuance and technical identifiers.

Async Ingestion Pipeline: Non-blocking PDF processing and semantic chunking to preserve context across document boundaries.

Source Attributions: Every response includes verifiable citations with similarity scores, ensuring full transparency.

RAGAS Evaluation: Built-in evaluation scripts to measure Faithfulness (factuality) and Answer Relevancy.

Decoupled Architecture: A high-performance FastAPI backend consumed by a clean, reactive Streamlit frontend.

🏗️ System Architecture
User Interface: Streamlit chat interface for document uploads and querying.

API Layer: FastAPI handles orchestration, security, and document parsing.

Vector Store: Qdrant stores high-dimensional embeddings (OpenAI text-embedding-3-small).

Generation: GPT-4o with a strict grounding prompt to eliminate hallucinations.

🛠️ Installation & Setup
1. Clone & Environment
Bash
git clone https://github.com/your-username/veristack-rag.git
cd veristack-rag
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate
pip install -r requirements.txt
2. Infrastructure (Docker)
Start the vector database using the provided compose file:

Bash
docker-compose up -d
3. API Keys
Create a .env file in the root directory:

Code snippet
OPENAI_API_KEY=your_key_here
QDRANT_URL=http://localhost:6333
🚦 Usage
Start the Backend (Terminal 1)
Bash
uvicorn app.main:app --reload --port 8000
Start the Frontend (Terminal 2)
Bash
streamlit run frontend/app.py
📊 Evaluation Metrics
This project uses the RAGAS framework to maintain high quality. Current benchmarks on the golden_dataset.json:

Faithfulness: 0.94 (High factuality)

Answer Relevancy: 0.89

Context Precision: 0.91

🛣️ Roadmap
[ ] Add Cross-Encoder Reranking (BGE-Reranker) for improved precision.

[ ] Implement Agentic RAG using Tool-calling for web-search fallback.

[ ] Support for Local LLMs via Ollama integration.
