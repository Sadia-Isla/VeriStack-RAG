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
        # Using the global Settings object (replaces deprecated ServiceContext)
        Settings.llm = OpenAI(model="gpt-4o", temperature=0.1)
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
        
        # Connection setup
        url = os.getenv("QDRANT_URL", "").strip("/")
        api_key = os.getenv("QDRANT_API_KEY")
        self.client = QdrantClient(url=url, api_key=api_key)
        self.vector_store = QdrantVectorStore(collection_name="docs", client=self.client)
    
    def process_pdf(self, file_path):
        """Extracts text from any text-based PDF without restrictive filtering."""
        clean_docs = []
        
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if not text:
                    continue
                
                # Basic cleanup: remove null bytes and normalize whitespace
                text = text.replace('\x00', '') 
                text = re.sub(r'\s+', ' ', text).strip()

                # Validation: Just ensures there is actual content
                if len(text) > 10:
                    clean_docs.append(Document(text=text))

        if not clean_docs:
            raise ValueError("No text found. If this is a scanned image, please use an OCR tool.")

        # Re-create collection for a fresh index
        if self.client.collection_exists("docs"):
            self.client.delete_collection("docs")
            
        self.client.create_collection(
            collection_name="docs",
            vectors_config={"size": 1536, "distance": "Cosine"}
        )

        # Indexing
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        VectorStoreIndex.from_documents(clean_docs, storage_context=storage_context)

    def query(self, text: str, top_k: int):
        """Retrieves and synthesizes answers using the vector store."""
        index = VectorStoreIndex.from_vector_store(self.vector_store)
        
        # 'tree_summarize' is best for synthesizing answers from multiple pages
        query_engine = index.as_query_engine(
            similarity_top_k=top_k, 
            response_mode="tree_summarize"
        )
        
        response = query_engine.query(text)
        
        # Format sources for UI display
        sources = [
            {
                "text": n.node.get_content()[:250] + "...", 
                "score": getattr(n, 'score', 0.0)
            } 
            for n in response.source_nodes
        ]
        
        return {"answer": str(response), "sources": sources}
