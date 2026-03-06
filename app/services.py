import os
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from qdrant_client import AsyncQdrantClient

class RAGEngine:
    def __init__(self):
        # 1. Update Settings to use the Key currently in the environment
        Settings.llm = OpenAI(model="gpt-4o", temperature=0.1)
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
        
        # 2. Connect to Qdrant Cloud
        self.client = AsyncQdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY") 
        )
        self.vector_store = QdrantVectorStore(collection_name="docs", aclient=self.client)
    
    async def process_pdf(self, file_path):
        from llama_index.core import SimpleDirectoryReader
        
        # Load and Index
        documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
        
        # Ensure the collection exists in Qdrant Cloud
        collection_exists = await self.client.collection_exists("docs")
        if not collection_exists:
            from qdrant_client.http import models as rest_models
            await self.client.create_collection(
                collection_name="docs",
                vectors_config=rest_models.VectorParams(size=1536, distance=rest_models.Distance.COSINE)
            )

        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        return VectorStoreIndex.from_documents(documents, storage_context=storage_context)

    async def query(self, text: str, top_k: int):
        # Connect to the existing cloud vector store
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
