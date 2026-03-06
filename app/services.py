class RAGEngine:
    def __init__(self):
        Settings.llm = OpenAI(model="gpt-4o", temperature=0.1)
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
        
        # Connect using both URL and API Key from environment
        self.client = AsyncQdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY") 
        )
        self.vector_store = QdrantVectorStore(collection_name="docs", aclient=self.client)
