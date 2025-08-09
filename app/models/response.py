#File: app/models/response.py
from __future__ import annotations

from typing import Optional, Dict, List, Any, Union
from pydantic import BaseModel, Field


class ScoredHit(BaseModel):
    """Elemento de resultado (chunk/documento) con score y metadata opcional."""
    id: Union[str, int]
    score: float
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class QueryResponse(BaseModel):
    """
    Respuesta de QA/búsqueda con:
      - answer: texto generado/seleccionado
      - hits: lista de candidatos (post re-ranking o directos del ANN si no hay re-ranking)
      - timings: métricas ms por etapa (encode/search/rerank/total)
      - params: parámetros efectivos usados (útil para auditoría)
      - warnings: observaciones opcionales
    """
    answer: str
    hits: List[ScoredHit] = Field(default_factory=list)

    # Parámetros efectivos
    candidates: int
    top_k: int
    rerank_applied: bool

    # Métricas/observabilidad
    timings: Optional[Dict[str, float]] = None  # {"encode_ms":..., "search_ms":..., "rerank_ms":..., "total_ms":...}
    params: Optional[Dict[str, Any]] = None     # {"model":..., "device":..., "ef":..., "exact":..., ...}
    warnings: Optional[List[str]] = None
