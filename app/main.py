
from fastapi import FastAPI, UploadFile, File
from app.services import RAGEngine
from app.models import QueryRequest, QueryResponse
import shutil

app = FastAPI(title="VeriStack RAG API")
engine = RAGEngine()

@app.post("/ingest")
async def ingest_doc(file: UploadFile = File(...)):
    temp_path = f"data/{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    await engine.process_pdf(temp_path)
    return {"status": "success", "filename": file.filename}

@app.post("/query", response_model=QueryResponse)
async def query_docs(request: QueryRequest):
    return await engine.query(request.query, request.top_k)
