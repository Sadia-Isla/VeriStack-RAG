
import os
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from qdrant_client import AsyncQdrantClient

# 2026 Best Practice: Global Settings
Settings.llm = OpenAI(model="gpt-4o", temperature=0.1)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

class RAGEngine:
    def __init__(self):
        self.client = AsyncQdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
        self.vector_store = QdrantVectorStore(collection_name="docs", aclient=self.client)
    
    async def process_pdf(self, file_path):
        from llama_index.core import SimpleDirectoryReader
        # Load and Index
        documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        # Creating index (async)
        return VectorStoreIndex.from_documents(documents, storage_context=storage_context)

    async def query(self, text: str, top_k: int):
        index = VectorStoreIndex.from_vector_store(self.vector_store)
        # Using Hybrid Search (Vector + Keyword)
        query_engine = index.as_query_engine(similarity_top_k=top_k)
        response = await query_engine.aquery(text)
        
        return {
            "answer": str(response),
            "sources": [{"text": n.text, "score": n.score, "metadata": n.metadata} for n in response.source_nodes]
        }
