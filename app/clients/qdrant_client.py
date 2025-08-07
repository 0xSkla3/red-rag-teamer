# File: app/clients/qdrant_client.py
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import PointStruct
from typing import List, Dict, Any, Optional, Tuple
from app.config import settings
from app.utils.logger import setup_logger
import time
import logging

logger = setup_logger(__name__)

class QdrantClientWrapper:
    def __init__(self):
        self.client = QdrantClient(
            url=settings.QDRANT_URL,
            prefer_grpc=settings.QDRANT_USE_GRPC,
            timeout=settings.QDRANT_TIMEOUT
        )
        self.collection_name = settings.RAG_COLLECTION
        self._ensure_collection()
        logger.info(f"Conectado a Qdrant en '{settings.QDRANT_URL}', colección '{self.collection_name}'")

    def _ensure_collection(self) -> None:
        """Crea la colección si no existe con configuración optimizada"""
        try:
            self.client.get_collection(self.collection_name)
        except (UnexpectedResponse, ValueError):
            hnsw_config = models.HnswConfigDiff(
                m=48,  # Mayor conectividad para precisión
                ef_construct=300,
                payload_indexing_threshold=100
            )
            
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=settings.EMBEDDING_DIM,
                    distance=models.Distance.COSINE,
                    on_disk=True
                ),
                hnsw_config=hnsw_config,
                optimizers_config=models.OptimizersConfigDiff(
                    indexing_threshold=0,
                    memmap_threshold=5000
                )
            )
            logger.info(f"Colección creada: '{self.collection_name}'")
            self._create_payload_indexes()

    def _create_payload_indexes(self) -> None:
        """Crea índices para campos técnicos basados en nuestra metadata"""
        index_fields = [
            "content_type", 
            "source_file",
            "tech_keywords"
        ]
        
        for field in index_fields:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD
                )
                logger.debug(f"Índice creado para: {field}")
            except Exception as e:
                logger.warning(f"Error creando índice para {field}: {str(e)}")

    def search(
        self, 
        embedding: List[float], 
        top_k: int,
        content_type: Optional[str] = None,
        source_file: Optional[str] = None,
        min_score: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Búsqueda optimizada con filtros para seguridad ofensiva
        
        Args:
            embedding: Vector de búsqueda
            top_k: Número de resultados
            content_type: Filtrar por tipo (exploit, code, manual, etc.)
            source_file: Filtrar por documento fuente
            min_score: Umbral mínimo de similitud
        """
        # Construir filtros
        must_conditions = []
        
        if content_type:
            must_conditions.append(
                models.FieldCondition(
                    key="content_type",
                    match=models.MatchValue(value=content_type)
            ))
        
        if source_file:
            must_conditions.append(
                models.FieldCondition(
                    key="source_file",
                    match=models.MatchValue(value=source_file)
            ))
        
        query_filter = models.Filter(must=must_conditions) if must_conditions else None
        
        # Búsqueda con parámetros optimizados
        start_time = time.time()
        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=embedding,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
            score_threshold=min_score,
            search_params=models.SearchParams(
                hnsw_ef=200,  # Mayor precisión
                exact=False
            )
        )
        
        # Procesar resultados
        results = []
        for hit in hits:
            results.append({
                "id": hit.id,
                "score": hit.score,
                "content": hit.payload.get("content", ""),
                "metadata": hit.payload.get("metadata", {})
            })
        
        latency = (time.time() - start_time) * 1000
        logger.info(f"Búsqueda retornó {len(results)} resultados en {latency:.2f}ms")
        return results

    def upsert_batch(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> Tuple[int, int]:
        """
        Upsert masivo optimizado para grandes volúmenes
        
        Args:
            documents: Lista de documentos con estructura:
                {
                    "id": str, 
                    "vector": List[float], 
                    "payload": {
                        "content": str,
                        "metadata": dict
                    }
                }
        """
        success = 0
        errors = 0
        total = len(documents)
        
        logger.info(f"Iniciando upsert masivo de {total} documentos")
        
        for i in range(0, total, batch_size):
            batch = documents[i:i+batch_size]
            points = [
                PointStruct(
                    id=doc["id"],
                    vector=doc["vector"],
                    payload=doc["payload"]
                ) for doc in batch
            ]
            
            try:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                    wait=False  # No esperar confirmación para mejor rendimiento
                )
                success += len(batch)
                logger.debug(f"Batch {i//batch_size+1}/{(total-1)//batch_size+1} enviado")
            except Exception as e:
                errors += len(batch)
                logger.error(f"Error en batch: {str(e)}")
                # Reintentar una vez
                try:
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=points,
                        wait=True
                    )
                    success += len(batch)
                    logger.info("Reintento exitoso")
                except Exception as retry_ex:
                    logger.error(f"Error en reintento: {str(retry_ex)}")
        
        # Sincronizar cambios
        self.client.update_collection(
            collection_name=self.collection_name,
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=20000,
                flush_interval_sec=10
            )
        )
        
        logger.info(f"Upsert masivo completado: {success} exitosos, {errors} fallidos")
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
                "config": {
                    "hnsw": collection_info.config.hnsw_config.dict(),
                    "optimizer": collection_info.config.optimizer_config.dict()
                }
            }
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas: {str(e)}")
            return {}