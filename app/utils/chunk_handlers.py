# app/utils/chunk_handlers.py
import re
import json
from abc import ABC, abstractmethod
from typing import List, Optional
from .chunk_strategies import IChunkStrategy

class ChunkHandler(ABC):
    def __init__(self, strategy: IChunkStrategy, next_handler: Optional['ChunkHandler'] = None):
        self.strategy = strategy
        self.next = next_handler

    @abstractmethod
    def can_handle(self, text: str) -> bool:
        pass

    def handle(self, text: str) -> List[str]:
        if self.can_handle(text):
            return self.strategy.chunk(text)
        elif self.next:
            return self.next.handle(text)
        else:
            return []

class HierarchicalChunkHandler(ChunkHandler):
    def can_handle(self, text: str) -> bool:
        patterns = getattr(self.strategy, 'header_patterns', [])
        return any(re.search(p, text) for p in patterns)

class CodeChunkHandler(ChunkHandler):
    def can_handle(self, text: str) -> bool:
        # Patrones para detectar múltiples lenguajes de programación
        code_patterns = [
            r'\b(def|function|class|struct|fn|let|var|const|import|package|include)\b',
            r'[{};=><\+\-\*/%&|\^!~]',  # Operadores comunes
            r'//|# |/\*|\*/',  # Comentarios
            r'\b(if|else|for|while|switch|case|return|break|continue)\b',
            r'\.\w+\s*\(',  # Llamadas a métodos
            r'^\s*\d+:\s*[a-zA-Z]',  # Etiquetas en assembly
            r'\b(section|global|extern|_start)\b'  # Directivas de ensamblador
        ]
        return any(re.search(pattern, text) for pattern in code_patterns)

class JsonChunkHandler(ChunkHandler):
    def can_handle(self, text: str) -> bool:
        text = text.strip()
        if (text.startswith('{') and text.endswith('}')) or (text.startswith('[') and text.endswith(']')):
            try:
                json.loads(text)
                return True
            except json.JSONDecodeError:
                return False
        return False

class LogChunkHandler(ChunkHandler):
    def __init__(self, strategy: IChunkStrategy, next_handler: Optional['ChunkHandler'] = None):
        super().__init__(strategy, next_handler)
        self.ts_regex = re.compile(
            r'(\[\d{2}/\w+/\d{4}:\d{2}:\d{2}:\d{2} [+-]\d{4}\])|'  # Nginx
            r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?)'  # ISO 8601
        )
    
    def can_handle(self, text: str) -> bool:
        sample_lines = text.split('\n')[:5]
        return any(self.ts_regex.search(line) for line in sample_lines)

class ExploitPayloadHandler(ChunkHandler):
    def can_handle(self, text: str) -> bool:
        exploit_patterns = [
            r'\b(shellcode|payload|exploit|ROP|gadget|buffer overflow)\b',
            r'\x[0-9a-fA-F]{2}',  # Bytes hexadecimales
            r'(\\x[0-9a-fA-F]{2})+',  # Secuencias de bytes
            r'\b(int 0x80|syscall)\b',  # Llamadas a sistema
            r'\b(CVE-\d+-\d+)\b',  # Identificadores CVE
            r'(0x)?[0-9a-fA-F]{8,}',  # Direcciones de memoria
            r'<\w+@\w+\.\w+>',  # Funciones en memoria
            r'\b(padding|nop sled|jmp esp|register)\b'
        ]
        return any(re.search(pattern, text) for pattern in exploit_patterns)

class AssemblyHandler(ChunkHandler):
    def can_handle(self, text: str) -> bool:
        assembly_patterns = [
            r'\b(mov|push|pop|call|ret|jmp|cmp|test|add|sub|xor|and|or)\b',
            r'[a-zA-Z0-9_]+:\s*;',  # Etiquetas
            r'\b(e?[a-ds]x|r[0-9]+d?|xmm[0-9]+|ymm[0-9]+|zmm[0-9]+)\b',  # Registros
            r'\b(dword ptr|qword ptr|byte ptr|word ptr)\b',
            r'\b(segment|offset|align)\b'
        ]
        return any(re.search(pattern, text) for pattern in assembly_patterns)

class BinaryDataHandler(ChunkHandler):
    def can_handle(self, text: str) -> bool:
        # Detectar datos binarios (no texto)
        if len(text) == 0:
            return False
            
        # Calcular ratio de caracteres no imprimibles
        non_printable = sum(1 for char in text if ord(char) < 32 and char not in '\n\r\t')
        ratio = non_printable / len(text)
        
        # Detectar secuencias binarias comunes
        hex_pattern = re.compile(r'(?:[0-9a-fA-F]{2}\s*)+')
        hex_sequences = hex_pattern.findall(text)
        hex_ratio = sum(len(seq) for seq in hex_sequences) / len(text) if text else 0
        
        return ratio > 0.3 or hex_ratio > 0.4

class TextChunkHandler(ChunkHandler):
    def can_handle(self, text: str) -> bool:
        return True