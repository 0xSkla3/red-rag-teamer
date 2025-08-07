# File: app/clients/qdrant_client.py
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue
from typing import List, Dict, Any, Optional, Tuple
from app.config import settings
from app.utils.logger import setup_logger
import time

logger = setup_logger(__name__)

class QdrantClientWrapper:
    def __init__(self):
        self.client = QdrantClient(
            url=settings.QDRANT_URL,
            prefer_grpc=False,
            timeout=settings.QDRANT_TIMEOUT
        )
        self.collection_name = settings.RAG_COLLECTION
        self._ensure_collection()
        logger.info(f"Connected to Qdrant at '{settings.QDRANT_URL}', collection '{self.collection_name}'")

    def _ensure_collection(self) -> None:
        """Crea la colección si no existe con configuración optimizada para seguridad informática"""
        try:
            # Verificar si la colección ya existe
            self.client.get_collection(self.collection_name)
        except (UnexpectedResponse, ValueError):
            # Crear colección con configuración optimizada
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=settings.EMBEDDING_DIM,
                    distance=models.Distance.COSINE,
                    on_disk=True  # Para grandes volúmenes
                ),
                optimizers_config=models.OptimizersConfigDiff(
                    indexing_threshold=0,  # Indexar inmediatamente
                    memmap_threshold=10000  # Usar memoria mapeada
                ),
                hnsw_config=models.HnswConfigDiff(
                    m=32,  # Mayor conectividad para precisión
                    ef_construct=200,  # Construcción más precisa
                    payload_indexing_threshold=50  # Indexar payloads pequeños
                )
            )
            logger.info(f"Created new collection: '{self.collection_name}'")
            
            # Crear índices de payload para búsquedas técnicas rápidas
            self._create_payload_indexes()

    def _create_payload_indexes(self) -> None:
        """Crea índices para campos técnicos comunes"""
        index_fields = [
            "doc_type", "platform", "cve", "risk_level", "is_technical"
        ]
        for field in index_fields:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD
                )
                logger.debug(f"Created payload index for: {field}")
            except Exception as e:
                logger.warning(f"Failed to create index for {field}: {str(e)}")

    def search(
        self, 
        embedding: List[float], 
        top_k: int,
        technical_only: bool = True,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Búsqueda con filtros para contenido técnico de seguridad
        
        Args:
            embedding: Vector de consulta
            top_k: Número de resultados
            technical_only: Filtrar solo contenido técnico
            filters: Filtros adicionales (ej: {"platform": "Windows"})
        """
        # Construir filtro compuesto
        must_conditions = []
        
        if technical_only:
            must_conditions.append(
                FieldCondition(key="is_technical", match=MatchValue(value=True))
        
        if filters:
            for key, value in filters.items():
                must_conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value)))
        
        query_filter = Filter(must=must_conditions) if must_conditions else None
        
        logger.debug(f"Searching with filter: {query_filter}")
        
        # Búsqueda con parámetros optimizados
        start_time = time.time()
        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=embedding,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
            search_params=models.SearchParams(
                hnsw_ef=128,  # Mayor precisión en recuperación
                exact=False
            )
        )
        
        # Procesar resultados
        results = []
        for hit in hits:
            result = {
                "id": hit.id,
                "score": hit.score,
                "content": hit.payload.get("content", ""),
                "metadata": hit.payload.get("metadata", {})
            }
            results.append(result)
        
        latency = (time.time() - start_time) * 1000
        logger.info(f"Search returned {len(results)} results in {latency:.2f}ms")
        return results

    def upsert(
        self, 
        ids: List[str],
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]]
    ) -> None:
        """Upsert con manejo de errores y reintentos"""
        points = [
            PointStruct(id=doc_id, vector=vec, payload=payload)
            for doc_id, vec, payload in zip(ids, vectors, payloads)
        ]
        
        # Intentar hasta 3 veces con backoff
        for attempt in range(3):
            try:
                operation_response = self.client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                    wait=True  # Esperar confirmación
                )
                if operation_response.status == models.UpdateStatus.COMPLETED:
                    logger.info(f"Upserted {len(points)} documents successfully")
                    return
                else:
                    logger.warning(f"Upsert partially failed: {operation_response.status}")
            except Exception as e:
                logger.error(f"Upsert attempt {attempt+1} failed: {str(e)}")
                time.sleep(1.5 ** attempt)  # Backoff exponencial
        
        logger.error(f"Failed to upsert {len(points)} documents after 3 attempts")

    def batch_upsert(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> Tuple[int, int]:
        """
        Upsert masivo optimizado para grandes volúmenes
        
        Args:
            documents: Lista de documentos con estructura:
                { "id": str, "vector": List[float], "payload": dict }
            batch_size: Tamaño de lote para upsert
        
        Returns:
            (success_count, error_count)
        """
        success = 0
        errors = 0
        total = len(documents)
        
        logger.info(f"Starting batch upsert of {total} documents in batches of {batch_size}")
        
        for i in range(0, total, batch_size):
            batch = documents[i:i+batch_size]
            try:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=[
                        PointStruct(
                            id=doc["id"],
                            vector=doc["vector"],
                            payload=doc["payload"]
                        ) for doc in batch
                    ],
                    wait=False  # No esperar confirmación para mejor rendimiento
                )
                success += len(batch)
                logger.debug(f"Submitted batch {i//batch_size+1}/{(total-1)//batch_size+1}")
            except Exception as e:
                errors += len(batch)
                logger.error(f"Batch upsert failed: {str(e)}")
        
        # Forzar sincronización al final
        self.client.update_collection(
            collection_name=self.collection_name,
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=10000,
                flush_interval_sec=5
            )
        )
        
        logger.info(f"Batch upsert completed: {success} succeeded, {errors} failed")
        return success, errors

    def get_collection_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de la colección para monitoreo"""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                "vectors_count": collection_info.vectors_count,
                "segments_count": len(collection_info.persisted_segments),
                "status": collection_info.status,
                "indexed_vectors": collection_info.indexed_vectors_count,
                "config": collection_info.config.dict()
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {str(e)}")
            return {}