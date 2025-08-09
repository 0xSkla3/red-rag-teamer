# File: app/utils/chunk_strategies.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Tuple, Callable
import re
import importlib
import numpy as np

from sentence_transformers import SentenceTransformer
from tree_sitter import Language, Parser

from app.utils.logger import setup_logger
from app.config import settings

logger = setup_logger(__name__, level=getattr(settings, "LOG_LEVEL", "INFO"))


# ========================= Interfaces base =========================

class ChunkStrategy(ABC):
    @abstractmethod
    def chunk(self, text: str) -> List[str]:
        """Devuelve una lista de chunks de texto."""
        raise NotImplementedError


# ========================= Agentic (embeddings) =========================

class AgenticChunking(ChunkStrategy):
    """
    Chunking 'inteligente' con embeddings para clasificar contenido:
      - exploit / code / manual / genérico
    Normalización activada (normalize=True) para que dot ≈ cosine.
    """

    def __init__(
        self,
        embedding_model_name: str = "BAAI/bge-large-en-v1.5",
        device: str = "cpu",
        min_chunk_size: int = 300,
        max_chunk_size: int = 1500,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        prompt_name: str = "document",
    ):
        self.device = device
        self.min_size = int(min_chunk_size)
        self.max_size = int(max_chunk_size)
        self.batch_size = int(batch_size)
        self.normalize = bool(normalize_embeddings)
        self.prompt_name = prompt_name

        self.model = SentenceTransformer(embedding_model_name, device=device)
        logger.info("AgenticChunking model loaded '%s' on %s", embedding_model_name, device)

        self.has_doc_prompt = (
            hasattr(self.model, "prompts")
            and isinstance(self.model.prompts, dict)
            and ("document" in self.model.prompts or "passage" in self.model.prompts)
        )

        self.reference_texts: Dict[str, str] = {
            "exploit": "Exploit code, shellcode, ROP, payload, vulnerability, CVE, buffer overflow, memory corruption.",
            "code":    "Source code, programming, function, class, import, include, def, struct, software development.",
            "manual":  "Technical manual, documentation, guide, section, chapter, header, tutorial, explanation.",
        }
        texts = list(self.reference_texts.values())
        embeddings = self._encode_doc(texts)
        self.ref_embeddings: Dict[str, np.ndarray] = {
            cat: embeddings[i] for i, cat in enumerate(self.reference_texts.keys())
        }
        logger.debug("AgenticChunking reference embeddings computed (%d)", len(texts))

    def _encode_doc(self, texts) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        if self.has_doc_prompt:
            pn = "document" if "document" in self.model.prompts else "passage"
            return self.model.encode(
                texts, prompt_name=pn, normalize_embeddings=self.normalize, convert_to_numpy=True
            )
        return self.model.encode(texts, normalize_embeddings=self.normalize, convert_to_numpy=True)

    def chunk(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []
        content_type = self._analyze_content_type(text)
        logger.debug("Agentic content type=%s", content_type)

        if content_type == "exploit":
            return self._chunk_exploits(text)
        elif content_type == "code":
            return self._chunk_code(text)
        elif content_type == "manual":
            return self._chunk_manual(text)
        else:
            return self._chunk_generic(text)

    def _analyze_content_type(self, text: str) -> str:
        sample = (text or "")[:1024]
        if not sample:
            return "generic"

        emb = self._encode_doc(sample)
        if isinstance(emb, np.ndarray) and emb.ndim == 2:
            emb = emb[0]

        best_cat = "generic"
        best_score = -1.0
        for cat, ref in self.ref_embeddings.items():
            try:
                score = float(np.dot(emb, ref))
            except Exception:
                score = -1.0
            if score > best_score:
                best_score = score
                best_cat = cat
        logger.debug("Agentic classify: %s (%.3f)", best_cat, best_score)
        return best_cat

    # ---- Estrategias por tipo ----

    def _chunk_exploits(self, text: str) -> List[str]:
        sections = re.split(r"(###\s*Technique:|###\s*Exploit Code:|###\s*Proof of Concept)", text)
        chunks: List[str] = []
        current = ""
        for section in sections:
            if not section:
                continue
            if len(current) + len(section) <= self.max_size:
                current = f"{current}{section}" if current else section
            else:
                if current:
                    chunks.append(current)
                current = section
        if current:
            chunks.append(current)
        return chunks

    def _chunk_code(self, text: str) -> List[str]:
        chunks: List[str] = []
        current = ""
        for line in text.split("\n"):
            if re.match(r"^\s*(def |class |function )", line) and current:
                chunks.append(current)
                current = line
            else:
                if len(current) + len(line) + 1 <= self.max_size:
                    current = (current + "\n" + line) if current else line
                else:
                    if current:
                        chunks.append(current)
                    current = line
        if current:
            chunks.append(current)
        return chunks

    def _chunk_manual(self, text: str) -> List[str]:
        chunks: List[str] = []
        current = ""
        for line in text.split("\n"):
            if re.match(r"^#+\s+", line) and current:
                chunks.append(current)
                current = line
            else:
                current = (current + "\n" + line) if current else line
            if len(current) >= self.max_size:
                chunks.append(current)
                current = ""
        if current:
            chunks.append(current)
        return chunks

    def _chunk_generic(self, text: str) -> List[str]:
        chunks: List[str] = []
        current = ""
        for paragraph in text.split("\n\n"):
            if not paragraph:
                continue
            if len(current) + len(paragraph) + 2 <= self.max_size:
                current = f"{current}\n\n{paragraph}" if current else paragraph
            else:
                if current:
                    chunks.append(current)
                current = paragraph
        if current:
            chunks.append(current)
        return chunks


# ========================= Jerárquico simple =========================

class HierarchicalChunkStrategy(ChunkStrategy):
    def __init__(self, header_patterns: Optional[List[str]] = None, min_chunk_size: int = 350):
        self.header_patterns = header_patterns or [r"^# ", r"^## ", r"^### "]
        self.compiled_patterns = [re.compile(p) for p in self.header_patterns]
        self.min_size = int(min_chunk_size)

    def chunk(self, text: str) -> List[str]:
        chunks: List[str] = []
        current = ""
        for line in text.split("\n"):
            if any(p.match(line) for p in self.compiled_patterns):
                if current and len(current) >= self.min_size:
                    chunks.append(current)
                    current = line
                else:
                    current = (current + "\n" + line) if current else line
            else:
                current = (current + "\n" + line) if current else line
        if current and len(current) >= self.min_size:
            chunks.append(current)
        return chunks


# ========================= AST con Tree-sitter (API 0.25) =========================

class AstChunkStrategy(ChunkStrategy):
    """
    AST con Tree-Sitter (>=0.25):
      - Usa paquetes por-lenguaje oficiales (p.ej. tree-sitter-python, ...).
      - Carga perezosa vía importlib; si un lenguaje no está instalado, devolvemos [text].
      - Regla de oro: no cortar dentro de funciones/clases; JSON se devuelve completo.
    """

    SUPPORTED = (
        "python","javascript","typescript","tsx","c","cpp","java","go","rust",
        "bash","php","ruby","c_sharp","json",
    )

    NODE_MAP: Dict[str, Tuple[str, ...]] = {
        "python": ("function_definition", "class_definition"),
        "javascript": ("function_declaration", "method_definition", "class_declaration"),
        "typescript": ("function_declaration", "method_signature", "class_declaration"),
        "tsx": ("function_declaration", "method_signature", "class_declaration"),
        "c": ("function_definition", "struct_specifier"),
        "cpp": ("function_definition", "class_specifier", "struct_specifier"),
        "java": ("method_declaration", "class_declaration"),
        "go": ("function_declaration", "method_declaration", "type_declaration"),
        "rust": ("function_item", "impl_item", "struct_item"),
        "bash": ("function_definition",),
        "php": ("function_declaration", "method_declaration", "class_declaration"),
        "ruby": ("method", "class"),
        "c_sharp": ("method_declaration", "class_declaration", "struct_declaration"),
        "json": (),  # JSON no se corta
    }

    # Mapa módulo -> callable que retorna el objeto de lenguaje; TS/TSX comparten paquete
    LANG_SPECS: Dict[str, Tuple[str, str]] = {
        "python":    ("tree_sitter_python", "language"),
        "javascript":("tree_sitter_javascript", "language"),
        "typescript":("tree_sitter_typescript", "language_typescript"),
        "tsx":       ("tree_sitter_typescript", "language_tsx"),
        "c":         ("tree_sitter_c", "language"),
        "cpp":       ("tree_sitter_cpp", "language"),
        "java":      ("tree_sitter_java", "language"),
        "go":        ("tree_sitter_go", "language"),
        "rust":      ("tree_sitter_rust", "language"),
        "bash":      ("tree_sitter_bash", "language"),
        "php":       ("tree_sitter_php", "language"),
        "ruby":      ("tree_sitter_ruby", "language"),
        "c_sharp":   ("tree_sitter_c_sharp", "language"),
        "json":      ("tree_sitter_json", "language"),
    }

    def __init__(self, min_size: int = 80, max_size: int = 10_000):
        self.min_size = int(min_size)
        self.max_size = int(max_size)
        self._parser_cache: Dict[str, Parser] = {}
        self._warned_langs: set[str] = set()

        logger.info("AstChunkStrategy: Tree-Sitter API 0.25 (Parser(language))")

    # ---------------- API ----------------

    def chunk(self, text: str) -> List[str]:
        if not text:
            return []

        lang = self._auto_detect_language(text)
        if not lang or lang == "json":
            return [text]

        parser = self._get_configured_parser(lang)
        if parser is None:
            return [text]

        try:
            tree = parser.parse(bytes(text, "utf-8", errors="ignore"))
        except Exception as e:
            self._warn_once(lang, f"parse() failed: {e}")
            return [text]

        target_nodes = self.NODE_MAP.get(lang, ())
        if not target_nodes:
            return [text]

        spans = self._collect_target_spans(tree.root_node, text, target_nodes)
        if not spans:
            return [text]

        chunks = self._assemble_chunks_from_spans(spans, text)
        return chunks or [text]

    # ---------------- helpers ----------------

    def _get_language_callable(self, lang: str) -> Optional[Callable[[], object]]:
        spec = self.LANG_SPECS.get(lang)
        if not spec:
            return None
        module_name, func_name = spec
        try:
            mod = importlib.import_module(module_name)
            fn = getattr(mod, func_name, None)
            if fn is None:
                raise AttributeError(f"{module_name}.{func_name} no encontrado")
            return fn
        except Exception as e:
            self._warn_once(lang, f"no se pudo importar {module_name} ({e})")
            return None

    def _get_configured_parser(self, lang: str) -> Optional[Parser]:
        if lang in self._parser_cache:
            return self._parser_cache[lang]
        fn = self._get_language_callable(lang)
        if fn is None:
            return None
        try:
            lang_obj = Language(fn())     # API moderna: construir Language desde callable del paquete
            parser = Parser(lang_obj)     # API 0.25: constructor recibe language
            self._parser_cache[lang] = parser
            return parser
        except Exception as e:
            self._warn_once(lang, f"no se pudo configurar parser para {lang} ({e})")
            return None

    def _warn_once(self, lang: str, msg: str):
        if lang not in self._warned_langs:
            logger.warning("AstChunkStrategy: %s", msg)
            self._warned_langs.add(lang)

    def _auto_detect_language(self, text: str) -> Optional[str]:
        lang = self._heuristic_lang(text)
        if lang:
            return lang

        best_lang = None
        best_count = 0
        for candidate in self.SUPPORTED:
            parser = self._get_configured_parser(candidate)
            if parser is None:
                continue
            try:
                tree = parser.parse(bytes(text, "utf-8", errors="ignore"))
                target_nodes = self.NODE_MAP.get(candidate, ())
                count = self._count_nodes(tree.root_node, target_nodes)
                if count > best_count:
                    best_count = count
                    best_lang = candidate
            except Exception as e:
                self._warn_once(candidate, f"parse autodetect failed: {e}")
                continue
        logger.debug("AstChunkStrategy autodetect: %s (nodes=%d)", best_lang, best_count)
        return best_lang

    def _heuristic_lang(self, text: str) -> Optional[str]:
        if text.startswith("#!/usr/bin/env python") or re.search(r"\bdef\s+\w+\(", text): return "python"
        if text.startswith("#!/bin/bash") or re.search(r"\bfunction\s+\w+\s*\(|\b\w+\s*\(\)\s*{", text): return "bash"
        if re.search(r"#include\s*<|int\s+main\s*\(", text): return "c"
        if re.search(r"using\s+namespace|std::", text): return "cpp"
        if re.search(r"\bpackage\s+main\b|\bfunc\s+\w+\s*\(", text): return "go"
        if re.search(r"\bclass\s+\w+|public\s+(class|static|void)", text): return "java"
        if re.search(r"\bfunction\b|\bclass\s+\w+|\bconsole\.", text): return "javascript"
        if re.search(r"\binterface\b|\bimplements\b|:\s*\w+\s*=>", text): return "typescript"
        if re.search(r"\bfn\s+\w+\s*\(", text): return "rust"
        return None

    def _count_nodes(self, root, target_types: Tuple[str, ...]) -> int:
        cnt = 0
        stack = [root]
        while stack:
            node = stack.pop()
            if node.type in target_types:
                cnt += 1
            stack.extend(node.children)
        return cnt

    def _collect_target_spans(self, root, text: str, target_types: Tuple[str, ...]) -> List[Tuple[int, int]]:
        spans: List[Tuple[int, int]] = []
        candidatas: List[Tuple[int, int]] = []

        stack = [root]
        while stack:
            node = stack.pop()
            if node.type in target_types:
                start = node.start_byte
                end = node.end_byte
                start = self._expand_leading_decorators(text, start)
                start = self._expand_leading_comments(text, start)
                candidatas.append((start, end))
            stack.extend(node.children)

        candidatas.sort(key=lambda x: x[0])
        for s, e in candidatas:
            if not spans:
                spans.append((s, e))
                continue
            ps, pe = spans[-1]
            if s <= pe:
                spans[-1] = (ps, max(pe, e))
            else:
                spans.append((s, e))
        return spans

    def _expand_leading_decorators(self, text: str, start: int) -> int:
        i = start
        while True:
            prev_line_start = text.rfind("\n", 0, i - 1) + 1 if i > 0 else 0
            if text[prev_line_start:i].strip().startswith("@"):
                i = prev_line_start
                continue
            break
        return i

    def _expand_leading_comments(self, text: str, start: int) -> int:
        i = start
        while True:
            prev_line_start = text.rfind("\n", 0, i - 1) + 1 if i > 0 else 0
            strip = text[prev_line_start:i].strip()
            if strip.startswith("#") or strip.startswith("//") or strip.startswith("/*"):
                i = prev_line_start
                continue
            break
        return i

    def _assemble_chunks_from_spans(self, spans: List[Tuple[int, int]], text: str) -> List[str]:
        chunks: List[str] = []
        if not spans:
            return chunks

        current_start, current_end = spans[0]
        for s, e in spans[1:]:
            candidate_len = (e - current_start)
            if candidate_len <= self.max_size and s <= current_end + 2048:
                current_end = e
            else:
                chunks.append(text[current_start:current_end])
                current_start, current_end = s, e

        chunks.append(text[current_start:current_end])

        header_imports = self._leading_import_block(text)
        if header_imports and chunks:
            chunks[0] = header_imports + "\n" + chunks[0]

        cleaned: List[str] = []
        buf = ""
        for ch in chunks:
            if len(ch) < self.min_size:
                buf = (buf + "\n" + ch) if buf else ch
            else:
                if buf:
                    merged = buf + "\n" + ch
                    if len(merged) <= self.max_size:
                        cleaned.append(merged)
                    else:
                        cleaned.append(buf)
                        cleaned.append(ch)
                    buf = ""
                else:
                    cleaned.append(ch)
        if buf:
            cleaned.append(buf)

        return cleaned

    def _leading_import_block(self, text: str) -> str:
        lines = text.splitlines()
        acc = []
        for line in lines[:200]:
            if re.match(r"^\s*(import |from |#include|using\s+namespace)", line):
                acc.append(line)
            elif line.strip() == "" and acc:
                acc.append(line)
            elif acc:
                break
        return "\n".join(acc).strip()
