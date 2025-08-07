# File: app/services/index_service.py
import anyio
import torch
import psutil
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
            device=settings.EMBEDDING_DEVICE,
            use_nli=True  # Siempre activar NLI para calidad técnica
        )
        self.qdrant = QdrantClientWrapper()
        self.batch_size = settings.EMBEDDING_BATCH_SIZE

    async def index_documents(self, docs: list[dict]) -> None:
        """Indexa documentos con gestión optimizada de batches"""
        total_docs = len(docs)
        ids = [doc.get('id', f"doc_{i}") for i, doc in enumerate(docs)]
        payloads = [doc.get('metadata', {}) for doc in docs]
        texts = [doc['content'] for doc in docs]

        # Procesamiento en batches con monitoreo de memoria
        for i in range(0, total_docs, self.batch_size):
            batch_ids = ids[i:i+self.batch_size]
            batch_texts = texts[i:i+self.batch_size]
            batch_payloads = payloads[i:i+self.batch_size]
            
            logger.debug(f"Processing batch {i//self.batch_size+1}/{(total_docs-1)//self.batch_size+1} ({len(batch_texts)} docs)")
            
            # Generar embeddings asíncronos con control de memoria
            embeddings = await self._safe_embed_batch(batch_texts)
            
            # Upsert en Qdrant
            self.qdrant.upsert(batch_ids, embeddings, batch_payloads)
            
            # Liberación explícita de memoria
            del embeddings
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Log de uso de recursos
            rss = psutil.Process().memory_info().rss / 1e6
            logger.debug(f"Indexed batch | Mem: {rss:.1f}MB")

    async def _safe_embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Genera embeddings con gestión de errores y memoria"""
        try:
            return await anyio.to_thread.run_sync(
                self.embedder.embed,
                texts
            )
        except torch.cuda.OutOfMemoryError:
            logger.warning("CUDA OOM! Reducing batch size and retrying...")
            self.batch_size = max(4, self.batch_size // 2)
            return await self._safe_embed_batch(texts)
        except RuntimeError as e:
            logger.error(f"Embedding failed: {str(e)}")
            return []