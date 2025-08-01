# File: app/services/rag_service.py
import anyio
from app.clients.qdrant_client import QdrantClientWrapper
from app.clients.llm_client import LLMClient
from app.clients.embedding_client import EmbeddingClient
from app.config import settings

class RAGService:
    def __init__(self):
        self.qdrant = QdrantClientWrapper()
        self.llm = LLMClient()
        self.embedder = EmbeddingClient(
            settings.EMBEDDING_MODEL,
            settings.EMBEDDING_DEVICE
        )

    async def answer(self, question: str, top_k: int = None) -> str:
        top_k = top_k or settings.TOP_K
        # 1) Obtener embedding
        embedding = await anyio.to_thread.run_sync(
            self.embedder.embed, question
        )

        # 2) Recuperar docs similares
        docs = self.qdrant.search(embedding, top_k)

        # 3) Construir prompt
        context = "\n---\n".join(docs)
        prompt = f"Contexto:\n{context}\n\nPregunta: {question}\nResponde de forma clara."

        # 4) Generar respuesta
        return await self.llm.generate(prompt)