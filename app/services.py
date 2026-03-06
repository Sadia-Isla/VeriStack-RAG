import os
import asyncio
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from qdrant_client import QdrantClient, AsyncQdrantClient

class RAGEngine:
    def __init__(self):
        # Configuration
        Settings.llm = OpenAI(model="gpt-4o", temperature=0.1)
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
        
        url = os.getenv("QDRANT_URL")
        api_key = os.getenv("QDRANT_API_KEY")

        # FIX: Provide both sync and async clients to avoid the AttributeError
        self.client = QdrantClient(url=url, api_key=api_key)
        self.aclient = AsyncQdrantClient(url=url, api_key=api_key)
        
        self.vector_store = QdrantVectorStore(
            collection_name="docs", 
            client=self.client, 
            aclient=self.aclient
        )
    
    async def process_pdf(self, file_path):
        from llama_index.core import SimpleDirectoryReader
        documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
        
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        # Using the simplified from_documents call
        return VectorStoreIndex.from_documents(documents, storage_context=storage_context)

    async def query(self, text: str, top_k: int):
        index = VectorStoreIndex.from_vector_store(self.vector_store)
        query_engine = index.as_query_engine(similarity_top_k=top_k)
        response = await query_engine.aquery(text)
        
        return {
            "answer": str(response),
            "sources": [
                {"text": n.text, "score": n.score if n.score else 0.0, "metadata": n.metadata} 
                for n in response.source_nodes
            ]
        }
