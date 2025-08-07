# File: app/utils/pdf_utils.py

import pdfplumber
import tracemalloc
import psutil
from typing import List, Dict
from app.utils.logger import setup_logger
from app.utils.chunk_strategies import ChunkStrategy

import fitz  # PyMuPDF


logger = setup_logger(__name__)

async def extract_chunks_from_pdf(
    pdf_path: str,
    strategy: ChunkStrategy
) -> List[Dict]:
    """
    Extrae y chunkea el PDF página a página usando la estrategia inyectada,
    devolviendo una lista de dicts con 'id', 'text' y 'meta' para cada chunk.
    Monitorea memoria para detectar fugas.
    """
    process = psutil.Process()
    tracemalloc.start()
    initial_rss = process.memory_info().rss / 1e6
    logger.info(f"🔍 Start extracting `{pdf_path}` | initial RSS: {initial_rss:.1f} MB")

    docs: List[Dict] = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            for page_no, page in enumerate(pdf.pages, start=1):
                rss_before = process.memory_info().rss / 1e6
                logger.info(f"📄 Page {page_no}/{total_pages} | RSS before: {rss_before:.1f} MB")

                # Extraer texto limpio de la página
                text = page.extract_text() or ""
                page.flush_cache()

                # Aplicar chunking según la estrategia inyectada
                for chunk_no, chunk in enumerate(strategy.chunk(text), start=1):
                    docs.append({
                        'id': f"{page_no:04d}-{chunk_no:04d}",  # ID numérico o UUID preferible
                        'text': chunk,
                        'meta': {
                            'source': pdf_path,
                            'page': page_no,
                            'chunk': chunk_no
                        }
                    })

                # Reporte de memoria post-chunking
                current, peak = tracemalloc.get_traced_memory()
                rss_after = process.memory_info().rss / 1e6
                logger.debug(
                    f"✅ Finished page {page_no} | RSS after: {rss_after:.1f} MB | "
                    f"tracemalloc current: {current/1e6:.1f} MB | peak: {peak/1e6:.1f} MB"
                )
                tracemalloc.clear_traces()

    finally:
        tracemalloc.stop()
        final_rss = process.memory_info().rss / 1e6
        logger.info(f"🏁 Extraction completed | total chunks: {len(docs)} | final RSS: {final_rss:.1f} MB")

    return docs


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extrae texto de un archivo PDF"""
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text