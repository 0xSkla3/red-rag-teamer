#File: app/models/request.py
from __future__ import annotations

from typing import Optional, Dict, Literal
from pydantic import BaseModel, Field, model_validator
from pydantic.alias_generators import to_pascal
from pydantic import AliasChoices

from app.config import settings


class QueryRequest(BaseModel):
    """
    Request de búsqueda/QA.
    - Acepta `question`, `query` o `q` como input del texto.
    - Permite override por-request de parámetros clave (candidatos ANN, top_k final, filtros, etc.).
    """
    # Aceptar múltiples nombres: "question", "query", "q"
    question: str = Field(
        ...,
        validation_alias=AliasChoices("question", "query", "q")
    )

    # Cuántos candidatos traigo del ANN antes de re-rankear
    candidates: int = Field(
        default_factory=lambda: settings.SEARCH_CANDIDATES,
        ge=1,
        description="Cantidad de candidatos que retorna la búsqueda vectorial (ANN) antes del re-ranking."
    )

    # Cuántos devuelvo al final (tras re-ranking si está activo)
    top_k: int = Field(
        default_factory=lambda: settings.RERANK_TOP_K,
        ge=1,
        description="Cantidad final de resultados devueltos."
    )

    # Filtros opcionales
    content_type: Optional[
        Literal["exploit", "code", "manual", "log", "text", "table", "generic"]
    ] = Field(
        default=None,
        description="Filtro por tipo de contenido (si está indexado en payload)."
    )
    source_file: Optional[str] = Field(
        default=None,
        description="Filtro por nombre de archivo fuente."
    )
    min_score: Optional[float] = Field(
        default=None,
        description="Umbral mínimo de similitud (Qdrant score_threshold)."
    )
    filters: Optional[Dict[str, str]] = Field(
        default=None,
        description="Filtros adicionales específicos del payload."
    )

    # Overrides de búsqueda/infra (opcional, si querés permitir por-request)
    rerank_enable: Optional[bool] = Field(
        default=None,
        description="Forzar on/off re-ranking (default: settings.RERANK_ENABLE)."
    )
    exact: Optional[bool] = Field(
        default=None,
        description="Forzar búsqueda exacta (full scan). Default: settings.QDRANT_EXACT."
    )
    ef: Optional[int] = Field(
        default=None,
        ge=1,
        description="Override de HNSW ef para esta consulta. Default: settings.QDRANT_HNSW_EF."
    )

    @model_validator(mode="after")
    def _coherence(self) -> "QueryRequest":
        """
        - Asegura top_k <= candidates.
        - Aplica defaults de flags si no vienen en el request.
        """
        if self.top_k > self.candidates:
            self.top_k = self.candidates

        if self.rerank_enable is None:
            self.rerank_enable = settings.RERANK_ENABLE
        if self.exact is None:
            self.exact = settings.QDRANT_EXACT
        if self.ef is None:
            self.ef = settings.QDRANT_HNSW_EF

        return self
