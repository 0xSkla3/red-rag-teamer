# File: app/config.py
from pydantic_settings import BaseSettings
import torch

class Settings(BaseSettings):
    OLLAMA_URL: str = "http://127.0.0.1:11434" #ollama
    QDRANT_URL: str = "http://127.0.0.1:6333"   #qdrant
    MODEL_NAME: str = "deepseek-r1:14b"
    HOST: str = "0.0.0.0"
    PORT: int = 5000
    TOP_K: int = 5
    EMBED_DIM: int = 768

    # Embedding config
    EMBEDDING_DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    RAG_COLLECTION: str = "documents"

    EMBEDDING_MODEL: str = "code-nli" #"nli-mpnet"  # nli-mpnet, nli-minilm, code-nli, "sentence-transformers/all-mpnet-base-v2"
    EMBEDDING_BATCH_SIZE = 64 if EMBEDDING_DEVICE == "cuda" else 16

    class Config:
        env_file = ".env"

settings = Settings()