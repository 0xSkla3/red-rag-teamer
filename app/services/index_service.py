import asyncio
import torch
import psutil
import uuid
from typing import List, Dict, Any, Optional

from app.clients.embedding_client import EmbeddingClient
from app.clients.qdrant_client import QdrantClientWrapper
from app.clients.reranker_client import RerankerClient
from app.config import settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__, level=getattr(settings, "LOG_LEVEL", "INFO"))


class IndexService:
    def __init__(self):
        self.embedder = EmbeddingClient(
            model_name=settings.EMBEDDING_MODEL,
            device=settings.EMBEDDING_DEVICE,
        )
        self.qdrant = QdrantClientWrapper()

        self.reranker: Optional[RerankerClient] = None
        if settings.RERANK_ENABLE:
            self.reranker = RerankerClient(
                model_name=settings.RERANK_MODEL,
                device=settings.RERANK_DEVICE,
                batch_size=settings.RERANK_BATCH_SIZE
            )
            logger.info("Re-ranking habilitado (%s)", settings.RERANK_MODEL)
        else:
            logger.info("Re-ranking deshabilitado")

        self.batch_size = int(settings.EMBEDDING_BATCH_SIZE)
        self.max_retries = 3

    async def index_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, int]:
        total_docs = len(documents)
        if total_docs == 0:
            logger.warning("No hay documentos para indexar")
            return {"success": 0, "errors": 0}

        logger.info("▶ Iniciando indexación de %d documentos...", total_docs)

        # 1) Embeddings (role=document)
        try:
            all_texts = [doc['content'] for doc in documents]
            all_embeddings = await self._embed_with_memory_management(all_texts, role="document")
        except Exception as e:
            logger.error("Error generando embeddings: %s", str(e))
            return {"success": 0, "errors": total_docs}

        if all_embeddings:
            emb_dim = len(all_embeddings[0])
            if emb_dim != int(settings.EMBEDDING_DIM):
                logger.warning(
                    "Dimensión de embedding (%d) != EMBEDDING_DIM (%d). "
                    "¿Colección creada con tamaño correcto?",
                    emb_dim, int(settings.EMBEDDING_DIM)
                )

        # 2) Preparar puntos para Qdrant (ID UUID v5 estable por (source_file,page,chunk))
        qdrant_docs = []
        for i, doc in enumerate(documents):
            pid = self._generate_point_id(doc['metadata'])
            qdrant_docs.append({
                "id": pid,  # string UUID v5
                "vector": all_embeddings[i],
                "payload": {
                    "content": doc['content'],
                    "metadata": doc['metadata']
                }
            })

        # liberar memoria GPU si aplica
        del all_embeddings
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 3) Upsert
        success, errors = self.qdrant.upsert_batch(qdrant_docs, batch_size=int(settings.QDRANT_BATCH_SIZE))
        logger.info("✓ Indexación completada: %d exitosos, %d errores", success, errors)
        return {"success": success, "errors": errors}

    async def _embed_with_memory_management(self, texts: List[str], role: str) -> List[List[float]]:
        embeddings: List[List[float]] = []
        original_batch = self.batch_size
        proc = psutil.Process()

        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i+self.batch_size]
            for attempt in range(self.max_retries):
                try:
                    batch_embeddings = await asyncio.to_thread(
                        self.embedder.embed, batch_texts, role
                    )
                    embeddings.extend(batch_embeddings)
                    rss = proc.memory_info().rss / 1e6
                    logger.debug(
                        "Embedding batch %d/%d | items=%d | RSS: %.1fMB",
                        i // self.batch_size + 1,
                        (len(texts) - 1) // self.batch_size + 1,
                        len(batch_texts),
                        rss
                    )
                    break
                except torch.cuda.OutOfMemoryError:
                    self.batch_size = max(1, self.batch_size // 2)
                    logger.warning("OOM! Reduciendo batch size a %d (intento %d)", self.batch_size, attempt + 1)
                    if attempt == self.max_retries - 1:
                        raise
            else:
                self.batch_size = original_batch

        return embeddings

    def _generate_point_id(self, metadata: Dict[str, Any]) -> str:
        """
        UUID v5 determinístico (sin coerción en el cliente):
        clave = source_file|page|chunk
        - Si cambia el contenido del mismo (source,page,chunk), el upsert actualiza el punto.
        """
        src   = str(metadata.get('source_file', 'unknown'))
        page  = str(metadata.get('page', '0'))
        chunk = str(metadata.get('chunk', metadata.get('chunk_index', '0')))
        key = f"{src}|{page}|{chunk}"
        return str(uuid.uuid5(uuid.NAMESPACE_URL, key))

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        content_type: Optional[str] = None,
        source_file: Optional[str] = None,
        min_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        query_vec = self.embedder.embed([query], role="query")[0]
        n_cand = int(settings.SEARCH_CANDIDATES)
        hits = self.qdrant.search(
            embedding=query_vec,
            top_k=n_cand,
            content_type=content_type,
            source_file=source_file,
            min_score=min_score,
            hnsw_ef=settings.QDRANT_HNSW_EF,
            exact=settings.QDRANT_EXACT
        )
        if not hits:
            return []

        final_k = top_k or int(settings.RERANK_TOP_K)
        if getattr(self, "reranker", None):
            docs = [h["content"] for h in hits]
            idx_order = self.reranker.rerank(query, docs, top_k=final_k)
            ranked = []
            for rank, idx in enumerate(idx_order, start=1):
                h = hits[idx]
                ranked.append({**h, "rank": rank, "rerank": True})
            return ranked

        hits = sorted(hits, key=lambda x: x["score"], reverse=True)[:final_k]
        for i, h in enumerate(hits, start=1):
            h["rank"] = i
            h["rerank"] = False
        return hits

    def get_collection_stats(self) -> Dict[str, Any]:
        return self.qdrant.get_collection_stats()
