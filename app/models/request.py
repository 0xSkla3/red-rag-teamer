# app/models/request.py
from pydantic import BaseModel
from app.config import settings

class QueryRequest(BaseModel):
    question: str
    top_k: int = settings.TOP_K
