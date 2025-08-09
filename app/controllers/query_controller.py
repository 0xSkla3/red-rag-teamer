# app/controllers/query_controller.py
from fastapi import APIRouter, HTTPException, Depends
from app.models.request import QueryRequest
from app.models.response import QueryResponse
from app.services.rag_service import RAGService

router = APIRouter()

# Lazy singleton (evita side effects al importar el módulo)
_service_instance: RAGService | None = None

def get_rag_service() -> RAGService:
    global _service_instance
    if _service_instance is None:
        _service_instance = RAGService()
    return _service_instance


@router.post("/", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest, service: RAGService = Depends(get_rag_service)):
    try:
        answer = await service.answer(req.question, req.top_k)
        return QueryResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
