#File: app/factories.py|
from __future__ import annotations

from typing import Optional
from app.utils.logger import setup_logger
from app.config import settings

from app.utils.chunk_strategies import (
    ChunkStrategy,
    AgenticChunking,
)
from app.utils.chunk_decorators import (
    TechnicalChunkOptimizer,
    LateChunkingDecorator,
    AstFallbackDecorator,
)

logger = setup_logger(__name__, level=getattr(settings, "LOG_LEVEL", "INFO"))


def build_strategy_pipeline(
    *,
    model_name: str,
    device: str,
    min_chunk_size: int,
    max_chunk_size: int,
    batch_size: int,
    late_threshold: float,
    use_ast: bool = False,
    normalize_embeddings: bool = True,
    prompt_name: str = "document",
) -> ChunkStrategy:
    """
    Construye el pipeline de chunking compuesto:
      [AST opcional (fallback)] -> AgenticChunking -> TechnicalChunkOptimizer -> LateChunkingDecorator

    - AST se aplica como *fallback*: si detecta código y produce cortes válidos, se usa;
      de lo contrario, cae al agentic.
    - LateChunking usa embeddings (ST v5) para fusionar semánticamente chunks adyacentes O(n).

    Args:
        model_name: modelo de embeddings para agentic/late (ej. BAAI/bge-large-en-v1.5)
        device: 'cpu' | 'cuda'
        min_chunk_size, max_chunk_size: límites de chunking
        batch_size: batch para embeddings internos (late/agentic)
        late_threshold: umbral de similitud para fusionar en late
        use_ast: si True, aplica AST como primera pasada (fallback)
        normalize_embeddings: normalizar embeddings (recomendado)
        prompt_name: 'document' o el que corresponda en ST v5

    Returns:
        ChunkStrategy compuesto listo para usar.
    """
    logger.info(
        "⛏️  Building chunk pipeline: model=%s device=%s min=%d max=%d batch=%d late_thr=%.3f ast=%s",
        model_name, device, min_chunk_size, max_chunk_size, batch_size, late_threshold, use_ast,
    )

    base: ChunkStrategy = AgenticChunking(
        embedding_model_name=model_name,
        device=device,
        min_chunk_size=min_chunk_size,
        max_chunk_size=max_chunk_size,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
        prompt_name=prompt_name,
    )

    if use_ast:
        # Intenta cortar con AST si conviene; si no, cae al agentic interno
        base = AstFallbackDecorator(
            wrapped=base,
            min_size=min_chunk_size,
            max_size=max_chunk_size,
        )

    tech = TechnicalChunkOptimizer(
        wrapped=base,
        min_size=min_chunk_size,
        max_size=max_chunk_size,
    )

    late = LateChunkingDecorator(
        wrapped=tech,
        model_name=model_name,
        device=device,
        similarity_threshold=late_threshold,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
        prompt_name=prompt_name,
    )

    logger.info("✅ Chunk pipeline listo")
    return late
