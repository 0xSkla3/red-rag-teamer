# File: app/config.py
from pydantic_settings import BaseSettings
import torch

class Settings(BaseSettings):
    OLLAMA_URL: str = "http://ollama:11434"
    QDRANT_URL: str = "http://qdrant:6333"
    MODEL_NAME: str = "deepseek-r1:14b"
    HOST: str = "0.0.0.0"
    PORT: int = 5000
    TOP_K: int = 5
    EMBED_DIM: int = 768

    # Embedding config
    EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"
    EMBEDDING_DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    RAG_COLLECTION: str = "documents"

    class Config:
        env_file = ".env"

settings = Settings()