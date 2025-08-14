#File: app/services/rag_service.py
from __future__ import annotations

import time
from typing import List, Tuple, Any

import anyio
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from app.clients.qdrant_client import QdrantClientWrapper
from app.clients.llm_client import LLMClient
from app.clients.embedding_client import EmbeddingClient
from app.config import settings


console = Console()


class RAGService:
    """
    Servicio RAG con detección automática de "roles" por modelo:

      role_mode ∈ {"prompt", "prefix", "none"}
        - prompt: el modelo expone prompts STv5 (query/document/passage).
        - prefix: E5 sin prompts -> se antepone 'query: ' a la consulta.
        - none:   sin roles (ej. all-mpnet-base-v2).

    Nota: La indexación (ingest) debería usar role="document" para consistencia.
    """

    def __init__(self):
        self.qdrant = QdrantClientWrapper()
        self.llm = LLMClient()
        self.embedder = EmbeddingClient(
            settings.EMBEDDING_MODEL,
            settings.EMBEDDING_DEVICE
        )
        self.model_name = settings.EMBEDDING_MODEL.lower()
        self.role_mode = self._infer_role_mode()

        console.log(
            Panel.fit(
                f"[bold]RAGService listo[/]\n"
                f"Modelo: [cyan]{settings.EMBEDDING_MODEL}[/]\n"
                f"Device: [magenta]{settings.EMBEDDING_DEVICE}[/]\n"
                f"Role mode: [green]{self.role_mode}[/]\n"
                f"Normalize: [yellow]{getattr(settings, 'ST_NORMALIZE', True)}[/]\n"
                f"Qdrant top_k default: [yellow]{settings.TOP_K}[/]",
                title="Boot",
            )
        )

    # ------------------------ PUBLIC API ------------------------

    async def answer(self, question: str, top_k: int | None = None) -> str:
        q = (question or "").strip()
        if not q:
            return "Pregunta vacía."

        k = int(top_k or settings.TOP_K or 5)
        k = max(1, min(k, 50))  # clamp básico

        # 1) Embed query
        t0 = time.perf_counter()
        query_vec = await anyio.to_thread.run_sync(self._embed_query, q)
        t1 = time.perf_counter()

        # 2) Retrieval
        docs = self.qdrant.search(query_vec, k)
        t2 = time.perf_counter()

        # 3) Preparar contexto (robusto según lo que devuelva Qdrant)
        texts, rows = self._extract_texts_and_rows(docs, limit=k)
        context = "\n---\n".join(texts) if texts else "(sin contexto)"

        # Logging bonita de los hits
        self._log_hits(rows, elapsed_embed=t1 - t0, elapsed_search=t2 - t1, k=k, q=q)

        # 4) LLM
        prompt = (
            f"Contexto:\n{context}\n\n"
            f"Pregunta: {q}\n"
            f"Responde de forma clara y concisa. Si el contexto es insuficiente, dilo."
        )
        answer = await self.llm.generate(prompt)
        t3 = time.perf_counter()

        console.log(f"[bold]LLM generado[/] en {t3 - t2:.3f}s")
        return answer

    # ----------------------- INTERNALS -------------------------

    def _infer_role_mode(self) -> str:
        """
        Detecta cómo tratar la query según el modelo:
          - STv5 con prompts -> 'prompt'
          - E5 sin prompts -> 'prefix'
          - Otro -> 'none'
        """
        has_doc_prompt = getattr(self.embedder, "has_doc_prompt", False)
        name = self.model_name

        if has_doc_prompt:
            # Modelos STv5 con prompts (p.ej., BGE/GTE en sbert >= 3.x)
            return "prompt"

        # Heurística para E5 (antiguos) sin prompts: requieren prefijos 'query: ...'
        if "e5" in name:
            return "prefix"

        # Algunos BGE viejos pueden funcionar sin prompts; no forzamos prefijo
        if "bge" in name or "gte" in name:
            return "none"

        # MPNet y genéricos
        return "none"

    def _embed_query(self, text: str) -> List[float]:
        """
        Devuelve un vector 1D listo para Qdrant.
        Aplica roles si corresponde al modelo.
        """
        if self.role_mode == "prompt":
            # STv5 con prompts -> role="query"
            vecs = self.embedder.embed(text, role="query")
        elif self.role_mode == "prefix":
            # E5 sin prompts -> anteponer prefijo textual
            vecs = self.embedder.embed(f"query: {text}", role="document")
        else:
            # Sin roles -> usar 'document' para mantener consistencia con indexación
            vecs = self.embedder.embed(text, role="document")

        if not vecs:
            return []
        return vecs[0]  # (dim,)

    def _extract_texts_and_rows(self, docs: Any, limit: int) -> Tuple[List[str], List[Tuple[str, float, str]]]:
        """
        Normaliza resultados de Qdrant a:
           - texts: List[str] para el contexto
           - rows:  List[(preview, score, source)] para logging
        Acepta:
           - List[str]
           - List[dict] con keys {'text'|'content', 'score'?, 'payload'?, 'meta'?}
           - List[object] con attrs .payload/.score/.id/.text
        """
        texts: List[str] = []
        rows: List[Tuple[str, float, str]] = []

        def _safe_get(d: Any, *keys, default=None):
            if isinstance(d, dict):
                for k in keys:
                    if k in d and d[k] is not None:
                        return d[k]
            return default

        for i, item in enumerate(docs or []):
            if i >= limit:
                break

            score = 0.0
            source = ""
            text = ""

            if isinstance(item, str):
                text = item
            elif isinstance(item, dict):
                text = _safe_get(item, "text", "content", default="")
                score = float(_safe_get(item, "score", default=0.0) or 0.0)
                payload = _safe_get(item, "payload", "meta", default={}) or {}
                source = payload.get("source_file") or payload.get("source") or payload.get("file", "")
                if not text and "payload" in item and isinstance(item["payload"], dict):
                    text = item["payload"].get("content", "") or item["payload"].get("text", "")
            else:
                # objeto devuelto por SDK (e.g., points)
                text = getattr(item, "text", "") or getattr(getattr(item, "payload", None), "text", "")
                score = float(getattr(item, "score", 0.0) or 0.0)
                payload = getattr(item, "payload", {}) or {}
                if isinstance(payload, dict):
                    source = payload.get("source_file") or payload.get("source") or payload.get("file", "")

            text = (text or "").strip()
            if not text:
                continue

            preview = (text[:120] + "…") if len(text) > 120 else text
            texts.append(text)
            rows.append((preview, score, source))

        return texts, rows

    def _log_hits(self, rows: List[Tuple[str, float, str]], *, elapsed_embed: float, elapsed_search: float, k: int, q: str) -> None:
        table = Table(title=f"RAG search (k={k}) | embed {elapsed_embed:.3f}s | search {elapsed_search:.3f}s")
        table.add_column("Top", justify="right", style="bold")
        table.add_column("Score", justify="right")
        table.add_column("Source", overflow="fold")
        table.add_column("Preview", overflow="fold")

        if not rows:
            console.log(Panel.fit(f"[yellow]Sin resultados[/]\nQuery: [italic]{q}[/]"))
            return

        for idx, (preview, score, src) in enumerate(rows, start=1):
            table.add_row(str(idx), f"{score:.4f}", src or "-", preview)

        console.print(table)
