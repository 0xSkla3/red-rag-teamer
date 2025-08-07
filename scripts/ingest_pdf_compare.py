# File: scripts/ingest_pdf_compare.py

import asyncio
import sys
from app.utils.pdf_utils import extract_chunks_from_pdf
from app.services.index_service import IndexService
from app.utils.chunk_strategies import SemanticChunkStrategy, TopicChunkStrategy
from app.config import settings

async def main(pdf_path: str):
    # Instancia servicios
    sem = SemanticChunkStrategy(
        model_name=settings.EMBEDDING_MODEL,
        device=settings.EMBEDDING_DEVICE,
        similarity_threshold=0.75
    )
    top = TopicChunkStrategy(num_topics=8, passes=15)

    # Extraer con cada estrategia
    docs_sem = await extract_chunks_from_pdf(pdf_path, sem)
    docs_top = await extract_chunks_from_pdf(pdf_path, top)

    # Evaluar “ratio” de chunks
    print(f"Semantic chunks: {len(docs_sem)}")
    print(f"Topic chunks:    {len(docs_top)}")
    print(f"Ratio sem/top:   {len(docs_sem)/len(docs_top):.2f}")

    # Indexar en diferentes colecciones si quieres compararlos
    indexer = IndexService()
    await indexer.index_documents(docs_sem)  # colección default
    # o variantemente cambiar settings.RAG_COLLECTION para topic

if __name__ == "__main__":
    pdf = sys.argv[1]
    asyncio.run(main(pdf))
