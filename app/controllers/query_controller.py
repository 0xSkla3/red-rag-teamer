# app/controllers/query_controller.py
from fastapi import APIRouter, HTTPException
from app.models.request import QueryRequest
from app.models.response import QueryResponse
from app.services.rag_service import RAGService

router = APIRouter()
service = RAGService()

@router.post("/", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    try:
        answer = await service.answer(req.question, req.top_k)
        return QueryResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
