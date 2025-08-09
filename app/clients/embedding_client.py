# File: app/clients/embedding_client.py
from __future__ import annotations

from typing import List, Union
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from app.utils.logger import setup_logger
from app.config import settings

logger = setup_logger(__name__, level=getattr(settings, "LOG_LEVEL", "INFO"))


class EmbeddingClient:
    """
    Cliente de embeddings basado en SentenceTransformer.

    - Sin kwargs no soportados (export/backend/provider) para máxima compatibilidad.
    - Normalización activable (STv5: dot ≈ cosine).
    - Usa prompts si el modelo los expone (document/passsage/query).
    - Devuelve siempre List[List[float]] para integración directa con el vector store.
    """

    def __init__(self, model_name: str, device: str):
        self.model_name = model_name
        pref = (device or "cpu").lower()
        self.device = "cuda" if (pref in ("cuda", "auto") and torch.cuda.is_available()) else "cpu"

        # Instanciación limpia (sin kwargs exóticos)
        self.model = SentenceTransformer(self.model_name, device=self.device)

        # Prompts STv5 si existen
        self.has_doc_prompt = (
            hasattr(self.model, "prompts")
            and isinstance(self.model.prompts, dict)
            and (("document" in self.model.prompts) or ("passage" in self.model.prompts) or ("query" in self.model.prompts))
        )

        self.batch_size = int(getattr(settings, "ST_ENCODE_BATCH_SIZE", 32) or 32)
        self.num_workers = int(getattr(settings, "ST_ENCODE_MP_WORKERS", 0) or 0)
        self.normalize = bool(getattr(settings, "ST_NORMALIZE", True))

        logger.info(
            "EmbeddingClient: loaded '%s' on %s (batch=%d, workers=%d, normalize=%s, has_doc_prompt=%s)",
            self.model_name, self.device, self.batch_size, self.num_workers, self.normalize, self.has_doc_prompt
        )

    # ---------------------- API pública ----------------------

    def embed(self, text: Union[str, List[str]], role: str = "document") -> List[List[float]]:
        """
        Genera embeddings para uno o varios textos.
        role: "document" (default) | "query"
        Retorna: List[List[float]] (N, dim)
        """
        texts: List[str] = [text] if isinstance(text, str) else list(text)
        if not texts:
            return []

        logger.debug(
            "Embedding %d %s(s) (avg len: %.0f chars)",
            len(texts), role, (sum(len(t) for t in texts) / max(len(texts), 1))
        )

        # Elegimos prompt si el modelo lo soporta
        prompt_name = None
        if self.has_doc_prompt:
            if role == "query":
                # preferimos 'query' y si no existe, caemos a 'document'/'passage'
                if "query" in self.model.prompts:
                    prompt_name = "query"
                elif "document" in self.model.prompts:
                    prompt_name = "document"
                elif "passage" in self.model.prompts:
                    prompt_name = "passage"
            else:
                # role=document → 'document' > 'passage'
                if "document" in self.model.prompts:
                    prompt_name = "document"
                elif "passage" in self.model.prompts:
                    prompt_name = "passage"

        embs: np.ndarray = self.model.encode(
            texts,
            prompt_name=prompt_name,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            show_progress_bar=False,
        )

        # aseguramos (N, dim)
        if embs.ndim == 1:
            embs = embs.reshape(1, -1)
        return embs.tolist()
