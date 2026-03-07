import os
import re
from llama_index.core import VectorStoreIndex, StorageContext, Settings, Document
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
        self.client = QdrantClient(url=url, api_key=api_key)
        self.vector_store = QdrantVectorStore(collection_name="docs", client=self.client)
    
    def process_pdf(self, file_path):
        # Use PyMuPDF for the cleanest text extraction
        import fitz 
        doc_pdf = fitz.open(file_path)
        clean_docs = []

        for page in doc_pdf:
            text = page.get_text("text")
            
            # STRICT FILTER: Remove XML, XMP Metadata, and Binary Junk
            if "<?xpacket" in text or "<rdf:Description" in text or "uuid:" in text:
                continue
            
            # Remove non-printable characters and extra whitespace
            clean_text = re.sub(r'[^\x20-\x7E\n]+', ' ', text)
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()

            # Only index if it looks like real sentences (alphabetic chars > 50%)
            letters = sum(c.isalpha() for c in clean_text)
            if len(clean_text) > 40 and (letters / len(clean_text)) > 0.4:
                clean_docs.append(Document(text=clean_text))

        if not clean_docs:
            raise ValueError("No readable text found in PDF. It might be an image-only scan.")

        # Wipe old data to ensure the 'junk' is gone
        if self.client.collection_exists("docs"):
            self.client.delete_collection("docs")
            
        from qdrant_client.http import models as rest_models
        self.client.create_collection(
            collection_name="docs",
            vectors_config=rest_models.VectorParams(size=1536, distance=rest_models.Distance.COSINE)
        )

        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        VectorStoreIndex.from_documents(clean_docs, storage_context=storage_context)

    def query(self, text: str, top_k: int):
        index = VectorStoreIndex.from_vector_store(self.vector_store)
        # Use 'tree_summarize' for summary requests to get better context
        query_engine = index.as_query_engine(
            similarity_top_k=top_k, 
            response_mode="tree_summarize" if "summary" in text.lower() else "compact"
        )
        
        response = query_engine.query(text)
        sources = [{"text": n.node.get_content()[:200] + "...", "score": getattr(n, 'score', 0.0)} for n in response.source_nodes]
        return {"answer": str(response), "sources": sources}
