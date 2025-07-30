# app/clients/qdrant_client.py
from qdrant_client import QdrantClient
from app.config import settings

class QdrantClientWrapper:
    def __init__(self):
        self.client = QdrantClient(
            url=settings.QDRANT_URL,
            prefer_grpc=False
        )

    def search(self, embedding: list[float], top_k: int) -> list[str]:
        hits = self.client.search(
            collection_name="documents",
            query_vector=embedding,
            limit=top_k,
            with_payload=True
        )
        return [hit.payload.get("text", "") for hit in hits]
