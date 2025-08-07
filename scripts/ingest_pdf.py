# File: scripts/ingest_pdf.py
import asyncio
import sys
import time
import psutil
import tracemalloc
from tqdm.auto import tqdm
import torch
import logging
import re

from app.factories import ChunkPipelineBuilder, ChunkStrategyFactory
from app.utils.chunk_handlers import (
    HierarchicalChunkHandler, CodeChunkHandler, 
    JsonChunkHandler, LogChunkHandler,
    ExploitPayloadHandler, AssemblyHandler,
    TextChunkHandler
)
from app.services.index_service import IndexService
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

def detect_language(text: str) -> str:
    """Detecta el lenguaje de programación basado en patrones clave"""
    language_patterns = {
        'python': [r'\b(def|class|import)\b', r'#.*'],
        'c': [r'#include', r'\b(int|char)\b\s+\w+\s*\(', r'\{.*\}'],
        'cpp': [r'#include', r'\b(class|namespace)\b', r'std::'],
        'java': [r'\b(public\s+class|import\s+java\.)', r'@Override'],
        'javascript': [r'\bfunction\b', r'console\.log', r'\.then\(', r'\{.*\}'],
        'rust': [r'\bfn\b', r'let\s+mut', r'\.unwrap\(\)', r'!$'],
        'assembly': [r'\b(mov|push|pop|call|ret|jmp|cmp)\b', r'\b(eax|ebx|ecx|edx)\b'],
        'bash': [r'^#!/bin/bash', r'\$(.*?)\s*=', r'if \[.*\]'],
        'powershell': [r'^#Requires', r'\$[a-z]+?\s*=', r'-eq\s*"'],
    }
    
    for lang, patterns in language_patterns.items():
        if any(re.search(pattern, text) for pattern in patterns):
            return lang
    return 'unknown'

def extract_tech_keywords(text: str) -> List[str]:
    """Extrae palabras clave técnicas del texto"""
    keywords = set()
    # Patrones para detectar elementos técnicos
    patterns = {
        'function': r'\b(def|function|fn)\s+(\w+)\s*\(',
        'class': r'\b(class|struct|interface)\s+(\w+)',
        'api_call': r'\.(get|post|put|delete|patch|update)\s*\(',
        'vulnerability': r'\b(CVE-\d+-\d+|XSS|SQLi|RCE|LFI|RFI|XXE|CSRF)\b',
        'crypto': r'\b(AES|RSA|SHA-\d+|HMAC|PBKDF2)\b',
        'protocol': r'\b(HTTP/\d\.\d|FTP|SSH|SSL|TLS|DNS|SMTP)\b',
    }
    
    for key, pattern in patterns.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                keywords.update([m for m in match if m])
            else:
                keywords.add(match)
    
    return list(keywords)

def build_chunk_pipeline():
    """Construye el pipeline de chunking con la cadena de responsabilidad"""
    builder = ChunkPipelineBuilder()
    
    # Crear estrategias
    hierarchical_strat = ChunkStrategyFactory.get_strategy(
        'hierarchical', 
        header_patterns=[r'^# ', r'^## ', r'^### ']
    )
    ast_strat = ChunkStrategyFactory.get_strategy('ast', language='auto')
    json_strat = ChunkStrategyFactory.get_strategy('json')
    log_strat = ChunkStrategyFactory.get_strategy('log')
    # Para exploits y payloads, usamos ventana deslizante con tamaño más pequeño
    exploit_strat = ChunkStrategyFactory.get_strategy('sliding', window_size=256, overlap=0.2)
    # Estrategia semántica para texto general, con umbral dinámico
    semantic_strat = ChunkStrategyFactory.get_strategy(
        'semantic', 
        model_name='all-MiniLM-L6-v2',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        similarity_threshold=0.75,
        dynamic_threshold=True
    )
    
    # Aplicar late chunking a la estrategia semántica
    late_semantic_strat = ChunkStrategyFactory.get_strategy(
        'late',
        wrapped=semantic_strat,
        context_model_name='all-mpnet-base-v2',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Configurar cadena de handlers (orden de mayor a menor prioridad)
    builder.add_handler(HierarchicalChunkHandler(hierarchical_strat))
    builder.add_handler(ExploitPayloadHandler(exploit_strat))
    builder.add_handler(AssemblyHandler(ast_strat))
    builder.add_handler(CodeChunkHandler(ast_strat))
    builder.add_handler(JsonChunkHandler(json_strat))
    builder.add_handler(LogChunkHandler(log_strat))
    builder.add_handler(TextChunkHandler(late_semantic_strat))
    
    return builder.build()

async def main(document_path: str):
    logger.info(f"▶ Iniciando ingesta de documento: {document_path}")
    
    # Construir el pipeline de chunking
    pipeline = build_chunk_pipeline()
    
    # Medición de memoria y tiempo
    tracemalloc.start()
    mem_before = psutil.Process().memory_info().rss / 1e6
    t0 = time.time()
    
    logger.info(f"▶ Procesando documento y generando chunks...")
    chunks = pipeline.process(document_path)
    
    mem_after = psutil.Process().memory_info().rss / 1e6
    current, peak = tracemalloc.get_traced_memory()
    logger.info(
        f"   → Generados {len(chunks)} chunks en {time.time()-t0:.1f}s | "
        f"Mem inicial {mem_before:.1f}MB, actual {mem_after:.1f}MB, pico {peak/1e6:.1f}MB"
    )
    tracemalloc.stop()
    
    # Preparar documentos para indexación con metadatos técnicos
    documents = []
    for i, chunk in enumerate(chunks):
        # Detectar tipo de contenido
        if any(kw in chunk.lower() for kw in ['shellcode', 'payload', 'exploit']):
            doc_type = 'exploit'
        elif detect_language(chunk) != 'unknown':
            doc_type = 'code'
        elif re.search(r'\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}', chunk):
            doc_type = 'log'
        else:
            doc_type = 'text'
        
        # Construir metadatos técnicos
        metadata = {
            'doc_type': doc_type,
            'language': detect_language(chunk),
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
        torch.cuda.empty_cache()  # Limpiar memoria de GPU si se usa
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