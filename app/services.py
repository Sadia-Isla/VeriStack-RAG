import os
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from qdrant_client import AsyncQdrantClient

class RAGEngine:
    def __init__(self):
        # 1. Update Settings INSIDE the engine so it picks up the current os.environ["OPENAI_API_KEY"]
        Settings.llm = OpenAI(model="gpt-4o", temperature=0.1)
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
        
        # 2. Connection to Qdrant
        # Note: 'localhost' won't work on Streamlit Cloud unless you have a Qdrant Cloud URL
        self.client = AsyncQdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
        self.vector_store = QdrantVectorStore(collection_name="docs", aclient=self.client)
    
    async def process_pdf(self, file_path):
        from llama_index.core import SimpleDirectoryReader
        documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        # The index uses the global Settings we updated in __init__
        return VectorStoreIndex.from_documents(documents, storage_context=storage_context)

    async def query(self, text: str, top_k: int):
        index = VectorStoreIndex.from_vector_store(self.vector_store)
        query_engine = index.as_query_engine(similarity_top_k=top_k)
        response = await query_engine.aquery(text)
        
        # Return as the dictionary format your app.py expects
        return {
            "answer": str(response),
            "sources": [
                {"text": n.text, "score": n.score if n.score else 0.0, "metadata": n.metadata} 
                for n in response.source_nodes
            ]
        }
