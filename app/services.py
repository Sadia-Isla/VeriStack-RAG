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
        Settings.llm = OpenAI(model="gpt-4o", temperature=0.1)
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
        
        url = os.getenv("QDRANT_URL", "").strip("/")
        api_key = os.getenv("QDRANT_API_KEY")
        self.client = QdrantClient(url=url, api_key=api_key)
        self.vector_store = QdrantVectorStore(collection_name="docs", client=self.client)
    
    def process_pdf(self, file_path):
        clean_docs = []
        
        # Use pdfplumber: The most reliable for visible text
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if not text:
                    continue
                
                # --- HEAVY FILTERING ---
                # 1. Remove obvious PDF structural junk
                junk_patterns = [
                    r'obj\s*<', r'endobj', r'stream', r'endstream', 
                    r'xref', r'trailer', r'startxref', r'%%EOF',
                    r'<?xpacket', r'<rdf:', r'uuid:', r'/Metadata'
                ]
                
                is_junk = any(re.search(p, text) for p in junk_patterns)
                if is_junk:
                    continue

                # 2. Clean symbols and normalize whitespace
                text = re.sub(r'[^\x20-\x7E\n]+', ' ', text)
                text = re.sub(r'\s+', ' ', text).strip()

                # 3. Validation: Must be long enough and contain mostly letters
                if len(text) > 60:
                    letters = sum(c.isalpha() for c in text)
                    if (letters / len(text)) > 0.5: # At least 50% letters
                        clean_docs.append(Document(text=text))

        if not clean_docs:
            raise ValueError("Document appears to be an image or contains no extractable text.")

        # Wipe and Re-create Collection to ensure no old junk survives
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
        # Use tree_summarize to force the LLM to synthesize an answer from all chunks
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
