import os
import re
import pdfplumber
from llama_index.core import VectorStoreIndex, StorageContext, Settings, Document
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from qdrant_client import QdrantClient

class RAGEngine:
    def __init__(self):
        # Configure global settings
        Settings.llm = OpenAI(model="gpt-4o", temperature=0.1)
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
        
        url = os.getenv("QDRANT_URL", "").strip("/")
        api_key = os.getenv("QDRANT_API_KEY")
        
        self.client = QdrantClient(url=url, api_key=api_key)
        self.vector_store = QdrantVectorStore(collection_name="docs", client=self.client)
    
    def process_pdf(self, file_path):
        clean_docs = []
        
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if not text:
                    continue
                
                # Cleanup and normalization
                text = text.replace('\x00', '') 
                text = re.sub(r'[^\x20-\x7E\n]+', ' ', text)
                text = re.sub(r'\s+', ' ', text).strip()

                if len(text) > 20:
                    clean_docs.append(Document(text=text))

        if not clean_docs:
            raise ValueError("Document appears to be an image or contains no extractable text.")

        # Reset Collection
        if self.client.collection_exists("docs"):
            self.client.delete_collection("docs")
            
        self.client.create_collection(
            collection_name="docs",
            vectors_config={"size": 1536, "distance": "Cosine"}
        )

        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        VectorStoreIndex.from_documents(clean_docs, storage_context=storage_context)

    def query(self, text: str, top_k: int):
        index = VectorStoreIndex.from_vector_store(self.vector_store)
        
        query_engine = index.as_query_engine(
            similarity_top_k=top_k, 
            response_mode="tree_summarize"
        )
        
        response = query_engine.query(text)
        
        # Format sources with strictly cleaned indentation
        sources = []
        for n in response.source_nodes:
            sources.append({
                "text": n.node.get_content()[:250] + "...",
                "score": getattr(n, 'score', 0.0)
            })
        
        return {"answer": str(response), "sources": sources}
