from __future__ import annotations

import os
import json
import torch
from typing import ClassVar, Optional, Dict

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import model_validator

# -----------------------------------------------------------------------------
# Perfiles por defecto (pueden ser overrideados por variables de entorno)
# -----------------------------------------------------------------------------
VM_PROFILE = {
    # Embeddings
    "EMBEDDING_MODEL": "sentence-transformers/all-mpnet-base-v2",
    "EMBEDDING_DIM": 768,
    "EMBEDDING_DEVICE": "cpu",
    "EMBEDDING_BATCH_SIZE": 32,

    # ST v5 backend
    "ST_BACKEND": "torch",     # torch|onnx|openvino
    "ST_PROVIDER": "",         # EP para ONNX/OpenVINO (opcional)
    "ST_EXPORT_BACKEND": False,
    "ST_NORMALIZE": True,
    "ST_ENCODE_BATCH_SIZE": 32,
    "ST_ENCODE_MP_WORKERS": 0,

    # Qdrant (indexado diferido por calidad)
    "QDRANT_INDEX_BUILD_MODE": "on_query",  # never | on_query | finalize | manual
    "QDRANT_HNSW_EF": 400,
    "QDRANT_EXACT": False,

    # Reranking
    "RERANK_ENABLE": True,
    "RERANK_MODEL": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "RERANK_DEVICE": "auto",
    "RERANK_BATCH_SIZE": 32,
    "SEARCH_CANDIDATES": 100,
    "RERANK_TOP_K": 20,

    # Chunking thresholds
    "MIN_CHUNK_SIZE": 400,
    "MAX_CHUNK_SIZE": 1800,
    "LATE_CHUNKING_THRESHOLD": 0.85,

    # PDF
    "PDF_PREFER_TABLES": True,
    "PDF_CARRY_CHARS": 400,
    "PDF_TABLE_SETTINGS_JSON": None,
}

HOST_PROFILE = {
    "EMBEDDING_MODEL": "BAAI/bge-large-en-v1.5",
    "EMBEDDING_DIM": 1024,
    "EMBEDDING_DEVICE": "auto",   # preferir cuda si existe
    "EMBEDDING_BATCH_SIZE": 32,

    "ST_BACKEND": "torch",
    "ST_PROVIDER": "",
    "ST_EXPORT_BACKEND": False,
    "ST_NORMALIZE": True,
    "ST_ENCODE_BATCH_SIZE": 32,
    "ST_ENCODE_MP_WORKERS": 0,

    "QDRANT_INDEX_BUILD_MODE": "on_query",
    "QDRANT_HNSW_EF": 600,
    "QDRANT_EXACT": False,

    "RERANK_ENABLE": True,
    "RERANK_MODEL": "BAAI/bge-reranker-large",
    "RERANK_DEVICE": "auto",
    "RERANK_BATCH_SIZE": 32,
    "SEARCH_CANDIDATES": 200,
    "RERANK_TOP_K": 40,

    "MIN_CHUNK_SIZE": 400,
    "MAX_CHUNK_SIZE": 1800,
    "LATE_CHUNKING_THRESHOLD": 0.82,

    "PDF_PREFER_TABLES": True,
    "PDF_CARRY_CHARS": 400,
    "PDF_TABLE_SETTINGS_JSON": None,
}

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _resolve_device(device: str) -> str:
    d = (device or "cpu").lower()
    if d == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if d != "cpu" and not torch.cuda.is_available():
        return "cpu"
    return d

def _infer_dim_from_model(model_name: str, fallback: int) -> int:
    name = (model_name or "").lower()
    if "all-mpnet-base-v2" in name:
        return 768
    if "bge-large-en-v1.5" in name:
        return 1024
    return fallback

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
class Settings(BaseSettings):
    # Perfil
    RAG_PROFILE: str = "vm"  # vm | host

    # Logging
    LOG_LEVEL: str = "INFO"

    # Ollama/LLM
    OLLAMA_URL: str = "http://127.0.0.1:11434"

    # Embeddings (base; se pisan por perfil/env en el validador)
    EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"
    EMBEDDING_DIM: int = 768
    EMBEDDING_DEVICE: str = "cpu"     # cpu|cuda|auto
    EMBEDDING_BATCH_SIZE: int = 32

    # Agentic chunking model (por defecto alineado con EMBEDDING_MODEL)
    AGENTIC_MODEL: str = "BAAI/bge-large-en-v1.5"

    # ST v5 backend
    ST_BACKEND: str = "torch"         # torch|onnx|openvino
    ST_PROVIDER: str = ""             # EP opcional (onnx/openvino)
    ST_EXPORT_BACKEND: bool = False
    ST_NORMALIZE: bool = True
    ST_ENCODE_BATCH_SIZE: int = 32
    ST_ENCODE_MP_WORKERS: int = 0

    # Qdrant
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_USE_GRPC: bool = False
    QDRANT_TIMEOUT: int = 60
    QDRANT_BATCH_SIZE: int = 100
    RAG_COLLECTION: str = "red_team_rag"

    # HNSW / consulta
    QDRANT_INDEX_BUILD_MODE: str = "on_query"  # never | on_query | finalize | manual
    QDRANT_HNSW_EF: int = 400
    QDRANT_EXACT: bool = False

    # Reranking
    RERANK_ENABLE: bool = True
    RERANK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANK_DEVICE: str = "auto"
    RERANK_BATCH_SIZE: int = 32
    SEARCH_CANDIDATES: int = 100
    RERANK_TOP_K: int = 20

    # Chunking
    MIN_CHUNK_SIZE: int = 400
    MAX_CHUNK_SIZE: int = 1800
    LATE_CHUNKING_THRESHOLD: float = 0.85

    # PDF extraction / tablas
    PDF_PREFER_TABLES: bool = True
    PDF_CARRY_CHARS: int = 400
    PDF_TABLE_SETTINGS_JSON: Optional[str] = None
    PDF_TABLE_SETTINGS: Dict = {}  # derivado (JSON parseado)

    # Ollama
    OLLAMA_MODEL_NAME: str = "deepseek-r1:14b-qwen-distill-q4_K_M"
    OLLAMA_URL: str = "http://localhost:11434"
    # Seguridad (constante)
    SECURITY_KEYWORDS: ClassVar[list] = [
        "exploit", "shellcode", "ROP", "CVE", "vulnerability",
        "payload", "bypass", "privilege", "escalation", "malware",
    ]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @model_validator(mode="after")
    def _apply_profile_and_validate(self):
        profile = (self.RAG_PROFILE or "vm").lower()
        cfg = VM_PROFILE if profile in ("vm", "cpu") else HOST_PROFILE

        # --- Embeddings ---
        self.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", cfg["EMBEDDING_MODEL"])
        self.EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", cfg["EMBEDDING_DIM"]))
        self.EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", cfg["EMBEDDING_DEVICE"])
        self.EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", cfg["EMBEDDING_BATCH_SIZE"]))

        # --- ST v5 ---
        self.ST_BACKEND = os.getenv("ST_BACKEND", cfg["ST_BACKEND"])
        self.ST_PROVIDER = os.getenv("ST_PROVIDER", cfg["ST_PROVIDER"])
        self.ST_EXPORT_BACKEND = (os.getenv("ST_EXPORT_BACKEND", str(cfg["ST_EXPORT_BACKEND"]))).lower() == "true"
        self.ST_NORMALIZE = (os.getenv("ST_NORMALIZE", str(cfg["ST_NORMALIZE"]))).lower() == "true"
        self.ST_ENCODE_BATCH_SIZE = int(os.getenv("ST_ENCODE_BATCH_SIZE", cfg["ST_ENCODE_BATCH_SIZE"]))
        self.ST_ENCODE_MP_WORKERS = int(os.getenv("ST_ENCODE_MP_WORKERS", cfg["ST_ENCODE_MP_WORKERS"]))

        # --- Qdrant ---
        self.QDRANT_INDEX_BUILD_MODE = os.getenv("QDRANT_INDEX_BUILD_MODE", cfg["QDRANT_INDEX_BUILD_MODE"])
        self.QDRANT_HNSW_EF = int(os.getenv("QDRANT_HNSW_EF", cfg["QDRANT_HNSW_EF"]))
        self.QDRANT_EXACT = (os.getenv("QDRANT_EXACT", str(cfg["QDRANT_EXACT"]))).lower() == "true"

        # --- Reranking ---
        self.RERANK_ENABLE = (os.getenv("RERANK_ENABLE", str(cfg["RERANK_ENABLE"]))).lower() == "true"
        self.RERANK_MODEL = os.getenv("RERANK_MODEL", cfg["RERANK_MODEL"])
        self.RERANK_DEVICE = os.getenv("RERANK_DEVICE", cfg["RERANK_DEVICE"])
        self.RERANK_BATCH_SIZE = int(os.getenv("RERANK_BATCH_SIZE", cfg["RERANK_BATCH_SIZE"]))
        self.SEARCH_CANDIDATES = int(os.getenv("SEARCH_CANDIDATES", cfg["SEARCH_CANDIDATES"]))
        self.RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", cfg["RERANK_TOP_K"]))

        # --- Chunking ---
        self.MIN_CHUNK_SIZE = int(os.getenv("MIN_CHUNK_SIZE", cfg["MIN_CHUNK_SIZE"]))
        self.MAX_CHUNK_SIZE = int(os.getenv("MAX_CHUNK_SIZE", cfg["MAX_CHUNK_SIZE"]))
        self.LATE_CHUNKING_THRESHOLD = float(os.getenv("LATE_CHUNKING_THRESHOLD", cfg["LATE_CHUNKING_THRESHOLD"]))

        # --- PDF ---
        self.PDF_PREFER_TABLES = (os.getenv("PDF_PREFER_TABLES", str(cfg["PDF_PREFER_TABLES"]))).lower() == "true"
        self.PDF_CARRY_CHARS = int(os.getenv("PDF_CARRY_CHARS", cfg["PDF_CARRY_CHARS"]))
        self.PDF_TABLE_SETTINGS_JSON = os.getenv("PDF_TABLE_SETTINGS_JSON", cfg["PDF_TABLE_SETTINGS_JSON"])
        if self.PDF_TABLE_SETTINGS_JSON:
            try:
                self.PDF_TABLE_SETTINGS = json.loads(self.PDF_TABLE_SETTINGS_JSON)
            except Exception:
                self.PDF_TABLE_SETTINGS = {}

        # --- Dispositivos ---
        self.EMBEDDING_DEVICE = _resolve_device(self.EMBEDDING_DEVICE)
        self.RERANK_DEVICE = _resolve_device(self.RERANK_DEVICE)

        # --- Ajustes derivados ---
        inferred = _infer_dim_from_model(self.EMBEDDING_MODEL, self.EMBEDDING_DIM)
        if inferred != self.EMBEDDING_DIM:
            self.EMBEDDING_DIM = inferred

        # Alinear AGENTIC_MODEL por defecto (overrideable por env)
        self.AGENTIC_MODEL = os.getenv("AGENTIC_MODEL", self.EMBEDDING_MODEL)

        # ---- Ollama ---
        self.OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", self.OLLAMA_MODEL_NAME)

        return self


settings = Settings()
