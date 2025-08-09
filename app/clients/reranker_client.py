#file: app/clients/reranker_client.py
from __future__ import annotations

from typing import List, Sequence
import numpy as np
from sentence_transformers import CrossEncoder

from app.utils.logger import setup_logger
from app.config import settings

logger = setup_logger(__name__, level=getattr(settings, "LOG_LEVEL", "INFO"))


class RerankerClient:
    """
    Wrapper simple para CrossEncoder (ST v5).
    """

    def __init__(self, model_name: str, device: str, batch_size: int = 32, max_length: int = 512):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.model = CrossEncoder(model_name, device=device, max_length=max_length)
        logger.info("Loaded CrossEncoder '%s' on '%s' (batch=%d)", model_name, device, batch_size)

    def rerank(self, query: str, docs: Sequence[str], top_k: int) -> List[int]:
        """
        Retorna los índices de docs ordenados por score desc, recortados a top_k.
        """
        if not docs:
            return []
        pairs = [(query, d) for d in docs]
        scores = self.model.predict(pairs, batch_size=self.batch_size, convert_to_numpy=True)
        order = np.argsort(-scores)  # desc
        top_idx = order[:top_k].tolist()
        return top_idx
