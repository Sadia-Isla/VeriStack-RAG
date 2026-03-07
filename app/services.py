import os
from llama_index.core import VectorStoreIndex, SummaryIndex, StorageContext, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from qdrant_client import QdrantClient

class RAGEngine:
    def __init__(self):
        # High-quality models
        Settings.llm = OpenAI(model="gpt-4o", temperature=0.1)
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
        Settings.chunk_size = 1024 # Larger chunks for better context
        
        url = os.getenv("QDRANT_URL", "").strip("/")
        api_key = os.getenv("QDRANT_API_KEY")

        self.client = QdrantClient(url=url, api_key=api_key)
        self.vector_store = QdrantVectorStore(collection_name="docs", client=self.client)
    
    def process_pdf(self, file_path):
        from llama_index.readers.file import PyMuPDFReader
        loader = PyMuPDFReader()
        documents = loader.load_data(file_path=file_path)
        
        # Clean metadata from documents to force LLM to look at text content
        for doc in documents:
            doc.metadata = {"file_name": os.path.basename(file_path)}

        if not self.client.collection_exists("docs"):
            from qdrant_client.http import models as rest_models
            self.client.create_collection(
                collection_name="docs",
                vectors_config=rest_models.VectorParams(size=1536, distance=rest_models.Distance.COSINE)
            )

        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        VectorStoreIndex.from_documents(documents, storage_context=storage_context)
        # Return a SummaryIndex for broad questions
        return SummaryIndex.from_documents(documents)

    def query(self, text: str, top_k: int):
        # Determine if the user is asking for a summary
        is_summary = any(word in text.lower() for word in ["summary", "summarize", "overview", "what is this about"])
        
        index = VectorStoreIndex.from_vector_store(self.vector_store)
        
        # If it's a summary request, use a wider context window
        query_engine = index.as_query_engine(
            similarity_top_k=top_k if not is_summary else 10,
            response_mode="compact" if not is_summary else "tree_summarize"
        )
        
        response = query_engine.query(text)
        
        sources = []
        for node in response.source_nodes:
            txt = node.node.get_content().strip()
            # Filter out non-alphanumeric "junk"
            if len([c for c in txt if c.isalnum()]) > 20:
                sources.append({
                    "text": txt[:400] + "...",
                    "score": getattr(node, 'score', 0.0)
                })

        return {"answer": str(response), "sources": sources}
