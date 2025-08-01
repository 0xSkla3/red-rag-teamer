# File: app/clients/qdrant_client.py
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from app.config import settings

class QdrantClientWrapper:
    def __init__(self):
        self.client = QdrantClient(
            url=settings.QDRANT_URL,
            prefer_grpc=False
        )
        self.collection = settings.RAG_COLLECTION

    def search(self, embedding: list[float], top_k: int) -> list[str]:
        hits = self.client.search(
            collection_name=self.collection,
            query_vector=embedding,
            limit=top_k,
            with_payload=True
        )
        return [hit.payload.get("text", "") for hit in hits]

    def upsert(self,
               ids: list[str],
               vectors: list[list[float]],
               payloads: list[dict]
    ) -> None:
        points = [
            PointStruct(id=doc_id, vector=vec, payload=payload)
            for doc_id, vec, payload in zip(ids, vectors, payloads)
        ]
        self.client.upsert(
            collection_name=self.collection,
            points=points
        )