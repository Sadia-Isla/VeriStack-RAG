import os
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from qdrant_client import QdrantClient

class RAGEngine:
    def __init__(self):
        Settings.llm = OpenAI(model="gpt-4o", temperature=0.1)
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
        
        url = os.getenv("QDRANT_URL", "").strip("/")
        api_key = os.getenv("QDRANT_API_KEY")

        # Use ONLY the sync client for maximum stability in Streamlit
        self.client = QdrantClient(url=url, api_key=api_key)
        
        self.vector_store = QdrantVectorStore(
            collection_name="docs", 
            client=self.client
        )
    
    def process_pdf(self, file_path):
        from llama_index.readers.file import PyMuPDFReader
        loader = PyMuPDFReader()
        documents = loader.load_data(file_path=file_path)
        
        if not self.client.collection_exists("docs"):
            from qdrant_client.http import models as rest_models
            self.client.create_collection(
                collection_name="docs",
                vectors_config=rest_models.VectorParams(size=1536, distance=rest_models.Distance.COSINE)
            )

        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        VectorStoreIndex.from_documents(documents, storage_context=storage_context)

    def query(self, text: str, top_k: int):
        index = VectorStoreIndex.from_vector_store(self.vector_store)
        query_engine = index.as_query_engine(similarity_top_k=top_k)
        
        # Synchronous query call
        response = query_engine.query(text)
        
        sources = []
        for node in response.source_nodes:
            # Simple check to filter out the "garbled" text you saw earlier
            raw_text = node.node.get_content()
            if any(c.isalnum() for c in raw_text):
                sources.append({
                    "text": raw_text[:500] + "...", 
                    "score": getattr(node, 'score', 0.0)
                })

        return {
            "answer": str(response),
            "sources": sources
        }
