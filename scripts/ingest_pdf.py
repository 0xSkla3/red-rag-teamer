#File: scripts/ingest_pdf.py
import os
import sys
import argparse

# --- Pre-parse profile flag (antes de cargar settings) ---
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("document_path", nargs="?")
_pre.add_argument("--profile", "-p", choices=["vm", "host", "cpu", "gpu"], default=None)
_pre_args, _unknown = _pre.parse_known_args()
if _pre_args.profile:
    os.environ["RAG_PROFILE"] = "vm" if _pre_args.profile in ("vm", "cpu") else "host"

# --- Resto de imports ---
import asyncio
import time
import psutil
import tracemalloc
import torch
import re

from rich.progress import Progress

from app.config import settings
from app.utils.pdf_utils import extract_chunks_from_pdf  # dual-pass (fitz + pdfplumber)
from app.utils.chunk_strategies import AgenticChunking
from app.utils.chunk_decorators import (
    AstFallbackDecorator,
    TechnicalChunkOptimizer,
    LateChunkingDecorator,
)
from app.services.index_service import IndexService
from app.utils.logger import setup_logger

logger = setup_logger(__name__, level=settings.LOG_LEVEL)

# ---------- toggles de features ----------
ENABLE_AST_FALLBACK = getattr(settings, "ENABLE_AST_FALLBACK", True)
# Por defecto NO fusionamos para preservar la info
ENABLE_TECHNICAL_OPTIMIZER = getattr(settings, "ENABLE_TECHNICAL_OPTIMIZER", False)
ENABLE_LATE_CHUNKING = getattr(settings, "ENABLE_LATE_CHUNKING", False)

LATE_CHUNKING_THRESHOLD = float(getattr(settings, "LATE_CHUNKING_THRESHOLD", 0.86))
EMBED_DEVICE_PREF = (getattr(settings, "EMBEDDING_DEVICE", "cpu") or "cpu").lower()
EMBED_MODEL = settings.EMBEDDING_MODEL
EMBED_DIM = settings.EMBEDDING_DIM
EMBED_BATCH = int(settings.EMBEDDING_BATCH_SIZE)
MIN_CH = int(settings.MIN_CHUNK_SIZE)
MAX_CH = int(settings.MAX_CHUNK_SIZE)

PDF_CARRY_CHARS = int(getattr(settings, "PDF_CARRY_CHARS", 0))
PDF_PREFER_TABLES = bool(getattr(settings, "PDF_PREFER_TABLES", True))
PDF_TABLE_SETTINGS = getattr(settings, "PDF_TABLE_SETTINGS", {}) or {}
QDRANT_BATCH_SIZE = int(getattr(settings, "QDRANT_BATCH_SIZE", 32)) or 32


def build_chunk_pipeline():
    """Construye el pipeline con device y features configurables (sin fallbacks de firma)."""
    device = "cuda" if (EMBED_DEVICE_PREF in ("cuda", "auto") and torch.cuda.is_available()) else "cpu"
    logger.info(f"▶ Using device for embeddings/chunking: {device}")

    # Base agentic (STv5, normalización activa: dot ≈ cosine)
    agentic = AgenticChunking(
        embedding_model_name=EMBED_MODEL,
        device=device,
        min_chunk_size=MIN_CH,
        max_chunk_size=MAX_CH,
        batch_size=EMBED_BATCH,
        normalize_embeddings=True,
        prompt_name="document",
    )

    pipeline = agentic

    # Híbrido AST → respeta fences / no corta funciones si hay parser
    if ENABLE_AST_FALLBACK:
        pipeline = AstFallbackDecorator(
            wrapped=pipeline,
            min_size=MIN_CH,
            max_size=MAX_CH,
        )

    # Reglas técnicas → fusiona pequeños relacionados (opcional)
    if ENABLE_TECHNICAL_OPTIMIZER:
        pipeline = TechnicalChunkOptimizer(
            wrapped=pipeline,
            min_size=MIN_CH,
            max_size=MAX_CH,
        )

    # LateChunking (merge semántico global) — por defecto OFF para preservar info
    if ENABLE_LATE_CHUNKING:
        # Firma única esperada: ver cabecera del archivo
        pipeline = LateChunkingDecorator(
            wrapped=pipeline,
            model_name=EMBED_MODEL,
            device=device,
            similarity_threshold=LATE_CHUNKING_THRESHOLD,
            batch_size=EMBED_BATCH,
            normalize_embeddings=True,
            prompt_name="document",
        )

    return pipeline


# --------- utilidades de metadatos ---------

_CODE_RX = re.compile(r"(#include\b)|\b(def|function|class|import)\b")

def detect_content_type(text: str) -> str:
    if re.search(r"\b(shellcode|payload|exploit|ROP|gadget|buffer overflow)\b", text, re.IGNORECASE):
        return "exploit"
    if _CODE_RX.search(text):
        return "code"
    if re.search(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}", text):
        return "log"
    if re.search(r"^# |^## |^### ", text):
        return "manual"
    return "text"


def extract_tech_keywords(text: str) -> list[str]:
    keywords = set()
    patterns = [
        r"\b(CVE-\d+-\d+|XSS|SQLi|RCE|LFI|RFI|XXE|CSRF)\b",
        r"\b(ROP|ASLR|DEP|shellcode|exploit|payload)\b",
        r"\b(AES|RSA|SHA-\d+|HMAC|PBKDF2)\b",
        r"\b(HTTP/\d\.\d|FTP|SSH|SSL|TLS|DNS|SMTP)\b",
        r"\b(AWS|Azure|GCP|S3|EC2|IAM)\b",
        r"\b(Active Directory|AD|Kerberos|NTLM|LDAP)\b",
    ]
    for pattern in patterns:
        for m in re.findall(pattern, text, re.IGNORECASE):
            if isinstance(m, tuple):
                for x in m:
                    if x:
                        keywords.add(x)
            else:
                keywords.add(m)
    return sorted(keywords)


async def main(document_path: str):
    logger.info(f"▶ Iniciando ingesta de documento: {document_path}")
    logger.info(
        "Embeddings: profile=%s model=%s dim=%s device_pref=%s (torch.cuda.is_available=%s)",
        getattr(settings, "RAG_PROFILE", "vm"),
        EMBED_MODEL,
        EMBED_DIM,
        EMBED_DEVICE_PREF,
        torch.cuda.is_available(),
    )

    # Construir pipeline de chunking
    pipeline = build_chunk_pipeline()

    # Medición memoria/tiempo
    tracemalloc.start()
    proc = psutil.Process()
    mem_before = proc.memory_info().rss / 1e6
    t0 = time.time()

    documents = []

    try:
        if document_path.lower().endswith(".pdf"):
            logger.info("Procesando archivo PDF (dual-pass: fitz + tablas pdfplumber)...")
            with Progress() as progress:
                page_chunks = await extract_chunks_from_pdf(
                    pdf_path=document_path,
                    strategy=pipeline,
                    carry_chars=PDF_CARRY_CHARS,
                    trim_carry_on_first_chunk=True,
                    progress=progress,
                    prefer_tables=PDF_PREFER_TABLES,
                    table_settings=PDF_TABLE_SETTINGS,
                    dedupe=True,
                )
            for item in page_chunks:
                txt = item["text"]
                meta = item["meta"]
                meta.update({
                    "content_type": detect_content_type(txt),
                    "tech_keywords": extract_tech_keywords(txt),
                    "chunk_size": len(txt),
                    "source_file": os.path.basename(document_path),
                })
                documents.append({"content": txt, "metadata": meta})
        else:
            logger.info("Procesando archivo de texto...")
            with open(document_path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
            chunks = pipeline.chunk(text)
            for i, ch in enumerate(chunks, start=1):
                meta = {
                    "content_type": detect_content_type(ch),
                    "tech_keywords": extract_tech_keywords(ch),
                    "chunk_size": len(ch),
                    "source_file": os.path.basename(document_path),
                    "page": None,
                    "chunk": i,
                }
                documents.append({"content": ch, "metadata": meta})
    finally:
        mem_after = proc.memory_info().rss / 1e6
        current, peak = tracemalloc.get_traced_memory()
        logger.info(
            "   → Preparados %d chunks en %.1fs | Mem inicial %.1fMB, actual %.1fMB, pico %.1fMB",
            len(documents),
            time.time() - t0,
            mem_before,
            mem_after,
            peak / 1e6,
        )
        tracemalloc.stop()

    # Indexar en Qdrant
    indexer = IndexService()
    logger.info("▶ Indexando en Qdrant...")
    total = len(documents)
    batch_size = QDRANT_BATCH_SIZE

    for i in range(0, total, batch_size):
        batch = documents[i : i + batch_size]
        t1 = time.time()
        await indexer.index_documents(batch)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        rss = proc.memory_info().rss / 1e6
        logger.debug(
            "Indexado batch %d/%d (%d chunks) en %.2fs | Mem %.1fMB",
            i // batch_size + 1,
            (total - 1) // batch_size + 1,
            len(batch),
            time.time() - t1,
            rss,
        )

    logger.info("✓ Indexación completada (%d chunks). ✨", total)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingesta de documentos en Qdrant con chunking técnico.")
    parser.add_argument("document_path", help="Ruta al documento (.pdf o .txt/.md)")
    parser.add_argument(
        "--profile",
        "-p",
        choices=["vm", "host", "cpu", "gpu"],
        help="Selecciona perfil de ejecución (vm/cpu o host/gpu).",
    )
    args = parser.parse_args()
    asyncio.run(main(args.document_path))
