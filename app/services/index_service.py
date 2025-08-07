# File: app/services/index_service.py
import asyncio
import torch
import psutil
import logging
import numpy as np
from app.clients.embedding_client import EmbeddingClient
from app.clients.qdrant_client import QdrantClientWrapper
from app.config import settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class IndexService:
    def __init__(self):
        self.embedder = EmbeddingClient(
            model_name=settings.EMBEDDING_MODEL,
            device=settings.EMBEDDING_DEVICE
        )
        self.qdrant = QdrantClientWrapper()
        self.batch_size = settings.EMBEDDING_BATCH_SIZE
        self.max_retries = 3

    async def index_documents(self, documents: list[dict]) -> dict:
        """Indexa documentos con gestión optimizada de batches y memoria"""
        total_docs = len(documents)
        if total_docs == 0:
            logger.warning("No hay documentos para indexar")
            return {"success": 0, "errors": 0}
        
        logger.info(f"▶ Iniciando indexación de {total_docs} documentos...")
        
        # Generar todos los embeddings primero (con gestión de memoria)
        try:
            all_texts = [doc['content'] for doc in documents]
            all_embeddings = await self._embed_with_memory_management(all_texts)
        except Exception as e:
            logger.error(f"Error generando embeddings: {str(e)}")
            return {"success": 0, "errors": total_docs}
        
        # Preparar documentos para Qdrant
        qdrant_docs = []
        for i, doc in enumerate(documents):
            qdrant_docs.append({
                "id": self._generate_doc_id(doc['metadata']),
                "vector": all_embeddings[i],
                "payload": {
                    "content": doc['content'],
                    "metadata": doc['metadata']
                }
            })
        
        # Liberar memoria de embeddings
        del all_embeddings
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Realizar upsert masivo
        success, errors = self.qdrant.upsert_batch(qdrant_docs, batch_size=settings.QDRANT_BATCH_SIZE)
        
        # Reporte final
        logger.info(f"✓ Indexación completada: {success} exitosos, {errors} errores")
        return {"success": success, "errors": errors}

    async def _embed_with_memory_management(self, texts: list[str]) -> list[list[float]]:
        """Genera embeddings con gestión de memoria y reintentos"""
        embeddings = []
        current_batch = []
        original_batch_size = self.batch_size
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i+self.batch_size]
            for attempt in range(self.max_retries):
                try:
                    batch_embeddings = await asyncio.to_thread(
                        self.embedder.embed, 
                        batch_texts
                    )
                    embeddings.extend(batch_embeddings)
                    current_batch = []
                    
                    # Reporte de progreso
                    processed = min(i + self.batch_size, len(texts))
                    rss = psutil.Process().memory_info().rss / 1e6
                    logger.debug(
                        f"Embedding batch {i//self.batch_size+1}: "
                        f"{processed}/{len(texts)} | Mem: {rss:.1f}MB"
                    )
                    break
                except torch.cuda.OutOfMemoryError:
                    # Reducir batch size y reintentar
                    self.batch_size = max(1, self.batch_size // 2)
                    logger.warning(
                        f"OOM! Reduciendo batch size a {self.batch_size} (intento {attempt+1})"
                    )
                    if attempt == self.max_retries - 1:
                        raise
            else:
                # Restaurar batch size original después de los reintentos
                self.batch_size = original_batch_size
        
        return embeddings

    def _generate_doc_id(self, metadata: dict) -> str:
        """Genera ID único basado en metadatos"""
        source = metadata.get('source_file', 'unknown')
        index = metadata.get('chunk_index', 0)
        return f"{source}_{index}"

    def search(
        self, 
        query: str, 
        top_k: int = 5,
        content_type: str = None,
        source_file: str = None,
        min_score: float = 0.7
    ) -> list[dict]:
        """Búsqueda optimizada con embeddings"""
        # Generar embedding para la consulta
        query_embedding = self.embedder.embed([query])[0]
        
        # Realizar búsqueda en Qdrant
        return self.qdrant.search(
            embedding=query_embedding,
            top_k=top_k,
            content_type=content_type,
            source_file=source_file,
            min_score=min_score
        )

    def get_collection_stats(self) -> dict:
        """Obtiene estadísticas de la colección"""
        return self.qdrant.get_collection_stats()