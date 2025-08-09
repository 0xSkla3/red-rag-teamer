# app/clients/llm_client.py
import httpx
from app.config import settings

class LLMClient:
    def __init__(self):
        self.url = settings.OLLAMA_URL
        self.model = settings.OLLAMA_MODEL_NAME

    async def generate(self, prompt: str) -> str:
        payload = {"model": self.model, "prompt": prompt}
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(f"{self.url}/api/generate", json=payload)
            resp.raise_for_status()
            data = resp.json()
        return data.get("completion", "")
