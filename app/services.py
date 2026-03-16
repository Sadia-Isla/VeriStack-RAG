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
        # Keep original settings
        Settings.llm = OpenAI(model="gpt-4o", temperature=0.1)
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
        
        url = os.getenv("QDRANT_URL", "").strip("/")
        api_key = os.getenv("QDRANT_API_KEY")
        self.client = QdrantClient(url=url, api_key=api_key)
        # Initialize the vector store here to keep the reference stable
        self.vector_store = QdrantVectorStore(collection_name="docs", client=self.client)
    
    def process_pdf(self, file_path):
        clean_docs = []
        
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if not text:
                    continue
                
                # --- MODIFIED FILTERING (Fixed to allow your PDF) ---
                # Removed structural junk patterns that were false-positives
                text = re.sub(r'[^\x20-\x7E\n]+', ' ', text)
                text = re.sub(r'\s+', ' ', text).strip()

                # Validation: Keep your original logic but ensure it's not empty
                if len(text) > 60:
                    letters = sum(c.isalpha() for c in text)
                    if (letters / len(text)) > 0.4: # Slightly lowered to 40% for forms
                        clean_docs.append(Document(text=text))

        if not clean_docs:
            raise ValueError("Document appears to be an image or contains no extractable text.")

        # Wipe and Re-create Collection
        if self.client.collection_exists("docs"):
            self.client.delete_collection("docs")
            
        self.client.create_collection(
            collection_name="docs",
            vectors_config={"size": 1536, "distance": "Cosine"}
        )

        # FIX: Explicitly define storage context with the vector store
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        
        # Create the index using the documents and the context
        VectorStoreIndex.from_documents(
            clean_docs, 
            storage_context=storage_context
        )

    def query(self, text: str, top_k: int):
        # Keep original query interface
        index = VectorStoreIndex.from_vector_store(self.vector_store)
        query_engine = index.as_query_engine(
            similarity_top_k=top_k, 
            response_mode="tree_summarize"
        )
        
        response = query_engine.query(text)
        sources = [
            {"text": n.node.get_content()[:250] + "...", "score": getattr(n, 'score', 0.0)} 
            for n in response.source_nodes
        ]
        return {"answer": str(response), "sources": sources}
