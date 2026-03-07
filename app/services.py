import os
import asyncio
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from qdrant_client import QdrantClient, AsyncQdrantClient

class RAGEngine:
    def __init__(self):
        # 1. Global Model Configuration
        Settings.llm = OpenAI(model="gpt-4o", temperature=0.1)
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
        
        # 2. Get Credentials from Environment
        url = os.getenv("QDRANT_URL", "").strip("/")
        api_key = os.getenv("QDRANT_API_KEY")

        # Initialize both Sync and Async clients
        self.client = QdrantClient(url=url, api_key=api_key)
        self.aclient = AsyncQdrantClient(url=url, api_key=api_key)
        
        self.vector_store = QdrantVectorStore(
            collection_name="docs", 
            client=self.client, 
            aclient=self.aclient
        )
    
    async def process_pdf(self, file_path):
        from llama_index.readers.file import PyMuPDFReader
        loader = PyMuPDFReader()
        documents = loader.load_data(file_path=file_path)
        
        # Ensure collection exists in Qdrant Cloud
        if not self.client.collection_exists("docs"):
            from qdrant_client.http import models as rest_models
            self.client.create_collection(
                collection_name="docs",
                vectors_config=rest_models.VectorParams(size=1536, distance=rest_models.Distance.COSINE)
            )

        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        VectorStoreIndex.from_documents(documents, storage_context=storage_context)

    async def query(self, text: str, top_k: int):
        index = VectorStoreIndex.from_vector_store(self.vector_store)
        query_engine = index.as_query_engine(similarity_top_k=top_k)
        response = await query_engine.aquery(text)
        
        # Extract sources safely
        sources = [
            {"text": node.node.get_content(), "score": getattr(node, 'score', 0.0)} 
            for node in response.source_nodes
        ]

        return {
            "answer": str(response),
            "sources": sources
        }
