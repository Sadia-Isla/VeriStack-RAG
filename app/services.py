import os
import re
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from qdrant_client import QdrantClient

class RAGEngine:
    def __init__(self):
        Settings.llm = OpenAI(model="gpt-4o", temperature=0.1)
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
        Settings.chunk_size = 512 # Smaller chunks for more precise retrieval
        
        url = os.getenv("QDRANT_URL", "").strip("/")
        api_key = os.getenv("QDRANT_API_KEY")
        self.client = QdrantClient(url=url, api_key=api_key)
        self.vector_store = QdrantVectorStore(collection_name="docs", client=self.client)
    
    def process_pdf(self, file_path):
        # Switching to a more robust reader for complex PDFs
        from llama_index.readers.file import PDFReader
        loader = PDFReader()
        documents = loader.load_data(file=file_path)
        
        clean_docs = []
        for doc in documents:
            # 1. Strip out non-ASCII/Garbage symbols using Regex
            text = doc.get_content()
            clean_text = re.sub(r'[^\x20-\x7E]+', ' ', text) # Keep only readable English chars
            
            # 2. Ignore chunks that look like XML/Metadata code
            if "<?xpacket" in clean_text or "<rdf:RDF" in clean_text or len(clean_text.strip()) < 50:
                continue
                
            doc.set_content(clean_text)
            doc.metadata = {"source": os.path.basename(file_path)}
            clean_docs.append(doc)

        if not self.client.collection_exists("docs"):
            from qdrant_client.http import models as rest_models
            self.client.create_collection(
                collection_name="docs",
                vectors_config=rest_models.VectorParams(size=1536, distance=rest_models.Distance.COSINE)
            )

        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        VectorStoreIndex.from_documents(clean_docs, storage_context=storage_context)

    def query(self, text: str, top_k: int):
        index = VectorStoreIndex.from_vector_store(self.vector_store)
        # Use 'refine' mode for summaries to ensure it re-reads for quality
        query_engine = index.as_query_engine(similarity_top_k=top_k, response_mode="refine")
        
        response = query_engine.query(text)
        
        sources = []
        for node in response.source_nodes:
            sources.append({
                "text": node.node.get_content()[:300] + "...",
                "score": getattr(node, 'score', 0.0)
            })

        return {"answer": str(response), "sources": sources}
