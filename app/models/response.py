# app/models/response.py
from pydantic import BaseModel

class QueryResponse(BaseModel):
    answer: str
