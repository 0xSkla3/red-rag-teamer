# File: scripts/ingest_pdf.py
import asyncio
import sys
import time
import psutil
import tracemalloc
import torch
import re
from app.config import settings
from app.utils.pdf_utils import extract_text_from_pdf  # Nueva importación
from app.utils.chunk_strategies import (
    AgenticChunking, 
    LateChunkingDecorator,
    TechnicalChunkOptimizer
)
from app.services.index_service import IndexService
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

def build_chunk_pipeline():
    """Construye el pipeline simplificado de chunking"""
    # Estrategia principal usando embeddings de alta calidad
    agentic = AgenticChunking(
        embedding_model_name=settings.EMBEDDING_MODEL,
        device=settings.EMBEDDING_DEVICE,
        min_chunk_size=settings.MIN_CHUNK_SIZE,
        max_chunk_size=settings.MAX_CHUNK_SIZE
    )
    
    # Optimización semántica
    late_optimized = LateChunkingDecorator(
        wrapped=agentic,
        context_model=settings.EMBEDDING_MODEL,
        device=settings.EMBEDDING_DEVICE,
        similarity_threshold=settings.LATE_CHUNKING_THRESHOLD
    )
    
    # Optimización técnica final
    return TechnicalChunkOptimizer(
        wrapped=late_optimized,
        min_size=settings.MIN_CHUNK_SIZE
    )

def detect_content_type(text: str) -> str:
    """Detección simplificada de tipo de contenido"""
    if re.search(r'\b(shellcode|payload|exploit|ROP|gadget|buffer overflow)\b', text, re.IGNORECASE):
        return 'exploit'
    elif re.search(r'\b(def |function |class |import |#include)\b', text):
        return 'code'
    elif re.search(r'\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}', text):
        return 'log'
    elif re.search(r'^# |^## |^### ', text):
        return 'manual'
    return 'text'

def extract_tech_keywords(text: str) -> list[str]:
    """Extrae palabras clave técnicas del texto"""
    keywords = set()
    patterns = [
        r'\b(CVE-\d+-\d+|XSS|SQLi|RCE|LFI|RFI|XXE|CSRF)\b',
        r'\b(ROP|ASLR|DEP|shellcode|exploit|payload)\b',
        r'\b(AES|RSA|SHA-\d+|HMAC|PBKDF2)\b',
        r'\b(HTTP/\d\.\d|FTP|SSH|SSL|TLS|DNS|SMTP)\b',
        r'\b(AWS|Azure|GCP|S3|EC2|IAM)\b',  # Cloud
        r'\b(Active Directory|AD|Kerberos|NTLM|LDAP)\b'  # AD
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        keywords.update(matches)
    
    return list(keywords)

async def main(document_path: str):
    logger.info(f"▶ Iniciando ingesta de documento: {document_path}")
    
    # Cargar contenido del documento
    if document_path.lower().endswith('.pdf'):
        logger.info("Procesando archivo PDF...")
        text = extract_text_from_pdf(document_path)
    else:
        logger.info("Procesando archivo de texto...")
        with open(document_path, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
    
    # Construir pipeline de chunking
    pipeline = build_chunk_pipeline()
    
    # Medición de memoria y tiempo
    tracemalloc.start()
    mem_before = psutil.Process().memory_info().rss / 1e6
    t0 = time.time()
    
    logger.info(f"▶ Generando chunks...")
    chunks = pipeline.chunk(text)
    
    mem_after = psutil.Process().memory_info().rss / 1e6
    current, peak = tracemalloc.get_traced_memory()
    logger.info(
        f"   → Generados {len(chunks)} chunks en {time.time()-t0:.1f}s | "
        f"Mem inicial {mem_before:.1f}MB, actual {mem_after:.1f}MB, pico {peak/1e6:.1f}MB"
    )
    tracemalloc.stop()
    
    # Preparar documentos para indexación
    documents = []
    for i, chunk in enumerate(chunks):
        # Construir metadatos
        metadata = {
            'content_type': detect_content_type(chunk),
            'tech_keywords': extract_tech_keywords(chunk),
            'chunk_size': len(chunk),
            'source_file': document_path.split('/')[-1],
            'chunk_index': i
        }
        
        documents.append({
            'content': chunk,
            'metadata': metadata
        })
    
    # Indexar los documentos
    indexer = IndexService()
    logger.info("▶ Indexando en Qdrant...")
    total = len(documents)
    
    # Indexar en lotes para mejor rendimiento
    batch_size = 32
    for i in range(0, total, batch_size):
        batch = documents[i:i+batch_size]
        t1 = time.time()
        await indexer.index_documents(batch)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        rss = psutil.Process().memory_info().rss / 1e6
        logger.debug(
            f"Indexado batch {i//batch_size+1}/{(total-1)//batch_size+1} "
            f"({len(batch)} chunks) en {time.time()-t1:.2f}s | Mem {rss:.1f}MB"
        )
    
    logger.info(f"✓ Indexación completada ({total} chunks). ✨")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python -m scripts.ingest_pdf ruta/al/documento")
        sys.exit(1)
    doc_path = sys.argv[1]
    asyncio.run(main(doc_path))