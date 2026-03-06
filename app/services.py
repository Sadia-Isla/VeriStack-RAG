    async def process_pdf(self, file_path):
        # Improved PDF Reader for cleaner text extraction
        from llama_index.readers.file import PyMuPDFReader
        
        # 1. Load the document
        loader = PyMuPDFReader()
        documents = loader.load_data(file_path=file_path)
        
        # 2. Check if the collection exists, create it if not
        collection_exists = await self.client.collection_exists("docs")
        if not collection_exists:
            from qdrant_client.http import models as rest_models
            await self.client.create_collection(
                collection_name="docs",
                vectors_config=rest_models.VectorParams(size=1536, distance=rest_models.Distance.COSINE)
            )

        # 3. Index the documents
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        return VectorStoreIndex.from_documents(documents, storage_context=storage_context)
