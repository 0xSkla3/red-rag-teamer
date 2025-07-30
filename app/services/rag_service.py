# app/services/rag_service.py
from app.clients.qdrant_client import QdrantClientWrapper
from app.clients.llm_client import LLMClient
from app.config import settings

class RAGService:
    def __init__(self):
        self.qdrant = QdrantClientWrapper()
        self.llm = LLMClient()

    async def answer(self, question: str, top_k: int) -> str:
        # 1) Obtener embedding (stub, reemplazar con modelo real)
        embedding = await self._get_embedding(question)

        # 2) Recuperar docs similares
        docs = self.qdrant.search(embedding, top_k)

        # 3) Construir prompt
        context = "\n---\n".join(docs)
        prompt = f"Contexto:\n{context}\n\nPregunta: {question}\nResponde de forma clara."

        # 4) Generar respuesta
        return await self.llm.generate(prompt)

    async def _get_embedding(self, text: str) -> list[float]:
        # TODO: implementar llamada a endpoint de embeddings
        return [0.0] * settings.EMBED_DIM
