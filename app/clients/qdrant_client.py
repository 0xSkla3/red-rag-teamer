from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import PointStruct
from typing import List, Dict, Any, Optional, Tuple, Union
from app.config import settings
from app.utils.logger import setup_logger
import time
import uuid
import re

logger = setup_logger(__name__, level=getattr(settings, "LOG_LEVEL", "INFO"))

_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$"
)

PointId = Union[int, str]


class QdrantClientWrapper:
    def __init__(self):
        self.client = QdrantClient(
            url=settings.QDRANT_URL,
            prefer_grpc=settings.QDRANT_USE_GRPC,
            timeout=settings.QDRANT_TIMEOUT
        )
        self.collection_name = settings.RAG_COLLECTION
        self._ensure_collection()
        logger.info("Conectado a Qdrant '%s', colección '%s'", settings.QDRANT_URL, self.collection_name)

    def _ensure_collection(self) -> None:
        try:
            self.client.get_collection(self.collection_name)
        except (UnexpectedResponse, ValueError):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=int(settings.EMBEDDING_DIM),
                    distance=models.Distance.COSINE,
                    on_disk=True
                ),
                hnsw_config=models.HnswConfigDiff(
                    m=48,
                    ef_construct=300,
                ),
                optimizers_config=models.OptimizersConfigDiff(
                    indexing_threshold=0,
                    memmap_threshold=5000
                )
            )
            logger.info("Colección creada: '%s'", self.collection_name)
            self._create_payload_indexes()

    def _create_payload_indexes(self) -> None:
        index_fields = ["content_type", "source_file", "tech_keywords"]
        for field in index_fields:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD
                )
                logger.debug("Índice creado para: %s", field)
            except Exception as e:
                logger.warning("Error creando índice para %s: %s", field, str(e))

    # --------- validación estricta (sin coerción) ---------

    def _is_valid_point_id(self, pid: Any) -> bool:
        if isinstance(pid, int):
            return pid >= 0
        if isinstance(pid, uuid.UUID):
            return True
        if isinstance(pid, str) and _UUID_RE.match(pid):
            return True
        return False

    def _assert_valid_point_id(self, pid: Any) -> None:
        if not self._is_valid_point_id(pid):
            raise ValueError(
                f"Point ID inválido: {pid!r}. Debe ser uint64 o UUID (str/uuid.UUID). "
                "Generá el ID correctamente en el indexador."
            )

    # ------------------- API de búsqueda / upsert -------------------

    def search(
        self,
        embedding: List[float],
        top_k: int,
        content_type: Optional[str] = None,
        source_file: Optional[str] = None,
        min_score: float = 0.0,
        hnsw_ef: Optional[int] = None,
        exact: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        must_conditions = []
        if content_type:
            must_conditions.append(models.FieldCondition(
                key="content_type",
                match=models.MatchValue(value=content_type)
            ))
        if source_file:
            must_conditions.append(models.FieldCondition(
                key="source_file",
                match=models.MatchValue(value=source_file)
            ))
        query_filter = models.Filter(must=must_conditions) if must_conditions else None

        search_params = models.SearchParams(
            hnsw_ef=(hnsw_ef if hnsw_ef is not None else settings.QDRANT_HNSW_EF),
            exact=(exact if exact is not None else settings.QDRANT_EXACT)
        )

        t0 = time.time()
        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=embedding,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
            score_threshold=min_score,
            search_params=search_params
        )

        results = []
        for h in hits:
            results.append({
                "id": h.id,
                "score": h.score,
                "content": h.payload.get("content", ""),
                "metadata": h.payload.get("metadata", {})
            })

        logger.info(
            "Búsqueda %d cand. en %.2fms (ef=%s exact=%s, min_score=%.2f)",
            len(results), (time.time() - t0) * 1000,
            search_params.hnsw_ef, search_params.exact, min_score
        )
        return results

    def upsert_batch(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> Tuple[int, int]:
        success = 0
        errors = 0
        total = len(documents)
        logger.info("Iniciando upsert masivo de %d documentos", total)

        for i in range(0, total, batch_size):
            batch = documents[i:i+batch_size]
            points: List[PointStruct] = []

            # Validación previa (rápida y con error claro)
            for doc in batch:
                self._assert_valid_point_id(doc["id"])
                points.append(PointStruct(
                    id=doc["id"],
                    vector=doc["vector"],
                    payload=doc["payload"]
                ))

            try:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                    wait=False
                )
                success += len(batch)
                logger.debug("Batch %d/%d enviado", i//batch_size+1, (total-1)//batch_size+1)
            except Exception as e:
                errors += len(batch)
                logger.error("Error en batch %d: %s", i//batch_size+1, str(e))
                # reintento síncrono
                try:
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=points,
                        wait=True
                    )
                    success += len(batch)
                    errors -= len(batch)
                    logger.info("Reintento exitoso (batch %d)", i//batch_size+1)
                except Exception as retry_ex:
                    logger.error("Error en reintento (batch %d): %s", i//batch_size+1, str(retry_ex))

        # optimizadores post-upsert (best-effort)
        try:
            self.client.update_collection(
                collection_name=self.collection_name,
                optimizers_config=models.OptimizersConfigDiff(
                    indexing_threshold=20000,
                    flush_interval_sec=10
                )
            )
        except Exception as e:
            logger.warning("No se pudo actualizar optimizadores post-upsert: %s", str(e))

        logger.info("Upsert masivo completado: %d exitosos, %d fallidos", success, errors)
        return success, errors

    def get_collection_stats(self) -> Dict[str, Any]:
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                "vectors_count": collection_info.vectors_count,
                "segments_count": len(collection_info.persisted_segments),
                "status": collection_info.status,
                "indexed_vectors": collection_info.indexed_vectors_count,
                "config": {
                    "hnsw": collection_info.config.hnsw_config.dict(),
                    "optimizer": collection_info.config.optimizer_config.dict()
                }
            }
        except Exception as e:
            logger.error("Error obteniendo estadísticas: %s", str(e))
            return {}
