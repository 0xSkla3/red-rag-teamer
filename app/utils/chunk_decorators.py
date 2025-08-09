#File: app/utils/chunk_decorators.py
from __future__ import annotations

from typing import List, Tuple, Optional
import re
import numpy as np

from sentence_transformers import SentenceTransformer

from app.utils.logger import setup_logger
from app.config import settings
from app.utils.chunk_strategies import ChunkStrategy, AstChunkStrategy

logger = setup_logger(__name__, level=getattr(settings, "LOG_LEVEL", "INFO"))


class LateChunkingDecorator(ChunkStrategy):
    """
    Optimización final de chunks basada en contexto semántico global (O(n)):
      - Embedding global del documento (resumen).
      - Embedding de snippets de cada chunk en batch.
      - Funde chunks adyacentes si sim >= threshold.
    """

    def __init__(
        self,
        wrapped: ChunkStrategy,
        model_name: str,
        device: str = "cpu",
        similarity_threshold: float = 0.82,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        prompt_name: str = "document",
    ):
        self.wrapped = wrapped
        self.threshold = float(similarity_threshold)
        self.batch_size = int(batch_size)
        self.normalize = bool(normalize_embeddings)
        self.prompt_name = prompt_name

        self.model = SentenceTransformer(model_name, device=device)
        self.has_doc_prompt = (
            hasattr(self.model, "prompts")
            and isinstance(self.model.prompts, dict)
            and ("document" in self.model.prompts or "passage" in self.model.prompts)
        )

        logger.info(
            "LateChunking loaded '%s' on %s (thr=%.3f, batch=%d, normalize=%s)",
            model_name, device, self.threshold, self.batch_size, self.normalize,
        )

    def _embed(self, texts) -> np.ndarray:
        # Forzar lista
        if isinstance(texts, str):
            texts = [texts]
        if self.has_doc_prompt:
            pn = "document" if "document" in self.model.prompts else "passage"
            return self.model.encode(
                texts, prompt_name=pn, normalize_embeddings=self.normalize, convert_to_numpy=True
            )
        else:
            return self.model.encode(
                texts, normalize_embeddings=self.normalize, convert_to_numpy=True
            )

    def chunk(self, text: str) -> List[str]:
        base_chunks = self.wrapped.chunk(text)
        if len(base_chunks) <= 1:
            return base_chunks

        # Embedding global (resumen de 2k chars)
        context = text[:2048]
        global_emb = self._embed(context)[0]

        # Embeddings de los siguientes chunks en batch (snippets)
        next_chunks = base_chunks[1:]
        snippets = [ch[:512] if ch else "" for ch in next_chunks]
        emb_mat = self._embed(snippets)  # shape: (N-1, dim)

        merged: List[str] = []
        current = base_chunks[0]

        for idx, next_chunk in enumerate(next_chunks):
            chunk_emb = emb_mat[idx]
            sim = float(np.dot(global_emb, chunk_emb))
            if sim >= self.threshold:
                current = f"{current}\n\n{next_chunk}"
            else:
                merged.append(current)
                current = next_chunk

        merged.append(current)
        return merged


class TechnicalChunkOptimizer(ChunkStrategy):
    """
    Reglas específicas para contenido de seguridad:
      - Fusiona chunks pequeños relacionados por keywords (ROP/ASLR/DEP/shellcode/exploit/CVE)
      - Añade etiquetas contextuales donde falte
    """

    def __init__(self, wrapped: ChunkStrategy, min_size: int = 400, max_size: int = 1800):
        self.wrapped = wrapped
        self.min_size = int(min_size)
        self.max_size = int(max_size)

    def chunk(self, text: str) -> List[str]:
        chunks = self.wrapped.chunk(text)
        optimized: List[str] = []
        current = ""

        for ch in chunks:
            if len(current) < self.min_size and self._are_related(current, ch):
                current = f"{current}\n\n{ch}" if current else ch
                if len(current) > self.max_size:
                    optimized.append(self._add_context(current))
                    current = ""
            else:
                if current:
                    optimized.append(self._add_context(current))
                current = ch

        if current:
            optimized.append(self._add_context(current))
        return optimized

    def _are_related(self, a: str, b: str) -> bool:
        kw_a = set(re.findall(r"\b(ROP|ASLR|DEP|shellcode|exploit|CVE-\d+-\d+)\b", a or "", re.IGNORECASE))
        kw_b = set(re.findall(r"\b(ROP|ASLR|DEP|shellcode|exploit|CVE-\d+-\d+)\b", b or "", re.IGNORECASE))
        return bool(kw_a and kw_b and (kw_a & kw_b))

    def _add_context(self, chunk: str) -> str:
        lc = (chunk or "").lower()
        if "shellcode" in lc and "platform" not in lc:
            return f"[Shellcode Technique]\n{chunk}"
        if "rop" in lc and not chunk.strip().lower().startswith("rop chain"):
            return f"ROP Technique: {chunk}"
        return chunk


class AstFallbackDecorator(ChunkStrategy):
    """
    Híbrido AST + wrapped:
      - Respeta bloques de código en Markdown (```lang ... ```).
      - AST para bloques de código (cuando hay grammar), wrapped para texto.
      - Nunca cortar dentro de función/clase; bloques de código pueden exceder max_size.
    """

    FENCE_RE = re.compile(
        r"(^```(?P<lang>[a-zA-Z0-9_+-]*)\s*$)(?P<code>.*?)(^```$)",
        re.MULTILINE | re.DOTALL,
    )

    def __init__(self, wrapped: ChunkStrategy, min_size: int = 300, max_size: int = 2000):
        self.wrapped = wrapped
        self.min_size = int(min_size)
        self.max_size = int(max_size)
        self.ast = AstChunkStrategy(min_size=min_size, max_size=max_size)

    def chunk(self, text: str) -> List[str]:
        if not text:
            return []

        if "```" not in text:
            ast_chunks = self.ast.chunk(text)
            if self._valuable(ast_chunks):
                logger.debug("AstFallback: AST directo (%d chunks)", len(ast_chunks))
                return ast_chunks
            logger.debug("AstFallback: fallback a wrapped (sin fences/AST no aporta)")
            return self.wrapped.chunk(text)

        segments = self._split_by_fences(text)
        final_chunks: List[str] = []

        for kind, payload, lang in segments:
            if kind == "code":
                if lang and self._lang_supported(lang):
                    code_chunks = self.ast.chunk(payload)
                    final_chunks.extend(code_chunks if code_chunks else [payload])
                else:
                    final_chunks.append(payload)
            else:
                text_chunks = self.wrapped.chunk(payload)
                final_chunks.extend(text_chunks)

        final_chunks = self._post_merge_small_texts(final_chunks)
        return final_chunks

    # ---------------- helpers ----------------

    def _split_by_fences(self, text: str) -> List[Tuple[str, str, Optional[str]]]:
        out: List[Tuple[str, str, Optional[str]]] = []
        last_end = 0
        for m in self.FENCE_RE.finditer(text):
            start, end = m.span()
            if start > last_end:
                out.append(("text", text[last_end:start], None))
            lang = (m.group("lang") or "").strip().lower() or None
            code = m.group("code")
            out.append(("code", code, lang))
            last_end = end
        if last_end < len(text):
            out.append(("text", text[last_end:], None))
        return out

    def _lang_supported(self, lang: str) -> bool:
        alias = {"py": "python", "js": "javascript", "ts": "typescript", "c++": "cpp",
                 "sh": "bash", "ps1": "bash", "c#": "c_sharp"}
        return alias.get(lang, lang) in AstChunkStrategy.SUPPORTED

    def _post_merge_small_texts(self, chunks: List[str]) -> List[str]:
        merged: List[str] = []
        buf = None
        for ch in chunks:
            if self._looks_like_code(ch):
                if buf:
                    merged.append(buf)
                    buf = None
                merged.append(ch)
                continue
            if buf is None:
                buf = ch
                continue
            if len(buf) < (self.min_size // 2) and len(ch) < (self.min_size // 2) and len(buf) + len(ch) <= self.max_size:
                buf = buf + "\n" + ch
            else:
                merged.append(buf)
                buf = ch
        if buf:
            merged.append(buf)
        return merged

    def _looks_like_code(self, block: str) -> bool:
        return bool(
            re.search(r"(\{|\};|#include|def\s+\w+\(|class\s+\w+|function\s+\w+\s*\(|std::|using\s+namespace)", block or "")
        )

    def _valuable(self, chunks: List[str]) -> bool:
        if not chunks:
            return False
        if len(chunks) <= 1:
            return len(chunks[0]) >= max(self.min_size, 160)
        return any(len(c) >= self.min_size for c in chunks)
