#app/pipeline.py
from typing import List
from .chunk_handlers import ChunkHandler

class BaseChunkPipeline:
    def __init__(self, handler_chain: ChunkHandler):
        self.handler_chain = handler_chain

    def process(self, document_path: str) -> List[str]:
        text = self.load(document_path)
        preproc = self.preprocess(text)
        chunks = self.detect_and_chunk(preproc)
        return self.evaluate(chunks)

    def load(self, path: str) -> str:
        # cargar contenido de archivo
        pass

    def preprocess(self, text: str) -> str:
        # limpieza y normalización opcional
        return text

    def detect_and_chunk(self, text: str) -> List[str]:
        return self.handler_chain.handle(text)

    def evaluate(self, chunks: List[str]) -> List[str]:
        # opcional: filtrar u ordenar resultados
        return chunks