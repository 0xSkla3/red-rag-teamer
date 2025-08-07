# File: app/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import ClassVar

import torch

class Settings(BaseSettings):
    OLLAMA_URL: str = "http://127.0.0.1:11434" #ollama

    # Modelos de embeddings (calidad premium)
    EMBEDDING_MODEL: str = "BAAI/bge-large-en-v1.5"
    EMBEDDING_DEVICE: str = "cuda"
    EMBEDDING_DIM: int = 1024
    EMBEDDING_BATCH_SIZE: int = 16
    
    # Modelo para chunking inteligente
    AGENTIC_MODEL: str = "BAAI/bge-large-en-v1.5"  # Usar el mismo modelo de embeddings
    
    # Qdrant
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_USE_GRPC: bool = False
    QDRANT_TIMEOUT: int = 60
    QDRANT_BATCH_SIZE: int = 100
    RAG_COLLECTION: str = "red_team_rag"
    
    # Configuración de chunking
    MIN_CHUNK_SIZE: int = 400
    MAX_CHUNK_SIZE: int = 1800
    
    # Configuración avanzada
    EMBEDDING_NLI: bool = True
    EMBEDDING_SIMILARITY_THRESHOLD: float = 0.85
    LATE_CHUNKING_THRESHOLD: float = 0.82
    
    # Configuración de seguridad
    SECURITY_KEYWORDS: ClassVar[list] = [
        "exploit", "shellcode", "ROP", "CVE", "vulnerability",
        "payload", "bypass", "privilege", "escalation", "malware"
    ]
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()


#    EMBEDDING_MODEL: str = "code-nli" #"nli-mpnet"  # nli-mpnet, nli-minilm, code-nli, "sentence-transformers/all-mpnet-base-v2"
