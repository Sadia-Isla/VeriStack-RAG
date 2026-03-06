
from pydantic import BaseModel
from typing import List, Dict

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

class SourceNode(BaseModel):
    text: str
    score: float
    metadata: Dict

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceNode]
