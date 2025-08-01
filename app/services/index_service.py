# File: app/services/index_service.py
import anyio
from app.clients.embedding_client import EmbeddingClient
from app.clients.qdrant_client import QdrantClientWrapper
from app.config import settings

class IndexService:
    def __init__(self):
        self.embedder = EmbeddingClient(
            settings.EMBEDDING_MODEL,
            settings.EMBEDDING_DEVICE
        )
        self.qdrant = QdrantClientWrapper()

    async def index_documents(self, docs: list[dict]) -> None:
        texts = [doc['text'] for doc in docs]
        ids = [doc.get('id', str(i)) for i, doc in enumerate(docs)]
        payloads = [
            {**doc.get('meta', {}), 'text': doc['text']}
            for doc in docs
        ]
        embeddings = await anyio.to_thread.run_sync(
            lambda: [self.embedder.embed(t) for t in texts]
        )
        self.qdrant.upsert(ids, embeddings, payloads)