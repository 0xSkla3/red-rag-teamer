# File: app/utils/chunk_strategies.py
from abc import ABC, abstractmethod
from typing import List, Union, Optional, Dict
import re
import json
import statistics
import fitz  # PyMuPDF
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer


class ChunkStrategy(ABC):
    @abstractmethod
    def chunk(self, text: str) -> List[str]:
        pass

class AgenticChunking(ChunkStrategy):
    """
    Chunking inteligente usando embeddings para determinar puntos óptimos
    """
    def __init__(self, embedding_model_name: str = "BAAI/bge-large-en-v1.5", 
                 device: str = "cuda", 
                 min_chunk_size: int = 300, 
                 max_chunk_size: int = 1500):
        self.device = device
        self.min_size = min_chunk_size
        self.max_size = max_chunk_size
        self.model = SentenceTransformer(embedding_model_name, device=device)
        
        # Textos de referencia para clasificación
        self.reference_texts = {
            "exploit": "Exploit code, shellcode, ROP, payload, vulnerability, CVE, buffer overflow, memory corruption",
            "code": "Source code, programming, function, class, import, include, def, struct, software development",
            "manual": "Technical manual, documentation, guide, section, chapter, header, tutorial, explanation"
        }
        # Precalcular embeddings de referencia
        self.ref_embeddings = {}
        texts = list(self.reference_texts.values())
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        for i, cat in enumerate(self.reference_texts.keys()):
            self.ref_embeddings[cat] = embeddings[i]

    def chunk(self, text: str) -> List[str]:
        # Paso 1: Análisis de tipo de contenido
        content_type = self._analyze_content_type(text)
        
        # Paso 2: Selección de estrategia basada en tipo de contenido
        if content_type == "exploit":
            return self._chunk_exploits(text)
        elif content_type == "code":
            return self._chunk_code(text)
        elif content_type == "manual":
            return self._chunk_manual(text)
        else:
            return self._chunk_generic(text)
    
    def _analyze_content_type(self, text: str) -> str:
        """Clasifica el contenido usando similitud semántica"""
        # Embed el texto (primeros 512 caracteres para eficiencia)
        emb = self.model.encode([text[:512]], convert_to_numpy=True)[0]
        
        # Calcular similitud con categorías de referencia
        best_cat = "generic"
        best_score = -1
        
        for cat, ref_emb in self.ref_embeddings.items():
            score = np.dot(emb, ref_emb) / (np.linalg.norm(emb) * np.linalg.norm(ref_emb))
            if score > best_score:
                best_score = score
                best_cat = cat
        
        return best_cat
    
    def _chunk_exploits(self, text: str) -> List[str]:
        """División especializada para exploits manteniendo técnicas completas"""
        # 1. Identificar secciones técnicas
        sections = re.split(r'(### Technique:|### Exploit Code:|### Proof of Concept)', text)
        
        # 2. Fusionar secciones relacionadas
        chunks = []
        current = ""
        for section in sections:
            if len(current) + len(section) < self.max_size:
                current += section
            else:
                if current: chunks.append(current)
                current = section
        if current: chunks.append(current)
        
        return chunks
    
    def _chunk_code(self, text: str) -> List[str]:
        """Chunking para código preservando funciones completas"""
        chunks = []
        current = ""
        lines = text.split('\n')
        
        for line in lines:
            # Detectar inicio de función/clase
            if re.match(r'^\s*(def |class |function )', line) and current:
                chunks.append(current)
                current = line
            else:
                if len(current) + len(line) < self.max_size:
                    current += '\n' + line
                else:
                    if current: chunks.append(current)
                    current = line
        if current: chunks.append(current)
        return chunks
    
    def _chunk_manual(self, text: str) -> List[str]:
        """Chunking jerárquico para manuales técnicos"""
        chunks = []
        current = ""
        lines = text.split('\n')
        
        for line in lines:
            if re.match(r'^#+ ', line) and current:
                chunks.append(current)
                current = line
            else:
                current += '\n' + line
        if current: chunks.append(current)
        return chunks
    
    def _chunk_generic(self, text: str) -> List[str]:
        """Fallback para contenido genérico"""
        chunks = []
        current = ""
        for paragraph in text.split('\n\n'):
            if len(current) + len(paragraph) < self.max_size:
                current += '\n\n' + paragraph if current else paragraph
            else:
                if current: chunks.append(current)
                current = paragraph
        if current: chunks.append(current)
        return chunks

class HierarchicalChunkStrategy(ChunkStrategy):
    """
    Chunking basado en estructura jerárquica (solo texto)
    """
    def __init__(
        self,
        header_patterns: List[str] = None,
        min_chunk_size: int = 350
    ):
        self.header_patterns = header_patterns or [r'^# ', r'^## ', r'^### ']
        self.compiled_patterns = [re.compile(p) for p in self.header_patterns]
        self.min_size = min_chunk_size

    def chunk(self, text: str) -> List[str]:
        return self._chunk_text(text)

    def _chunk_text(self, text: str) -> List[str]:
        chunks = []
        current = ""
        lines = text.split('\n')
        
        for line in lines:
            if any(p.match(line) for p in self.compiled_patterns):
                if current and len(current) >= self.min_size:
                    chunks.append(current)
                    current = line
                else:
                    current += '\n' + line if current else line
            else:
                current += '\n' + line if current else line
        if current and len(current) >= self.min_size:
            chunks.append(current)
        return chunks

class AstChunkStrategy(ChunkStrategy):
    """
    Chunking basado en AST para múltiples lenguajes con enfoque en seguridad.
    Soporta: Python, C/C++, Assembly, PowerShell, JavaScript.
    """
    def __init__(self, languages: List[str] = None):
        self.languages = languages or ["python", "c", "cpp", "asm", "powershell", "javascript"]
        self.parsers = self._init_parsers()
    
    def _init_parsers(self) -> Dict:
        try:
            from tree_sitter import Language, Parser
            parser_map = {}
            for lang in self.languages:
                try:
                    language = Language(f'build/{lang}.so', lang)
                    parser = Parser()
                    parser.set_language(language)
                    parser_map[lang] = parser
                except:
                    continue
            return parser_map
        except ImportError:
            return {}

    def chunk(self, text: str) -> List[str]:
        lang = self._detect_language(text)
        if lang not in self.parsers:
            return [text]
        
        tree = self.parsers[lang].parse(bytes(text, "utf-8"))
        chunks = []
        self._extract_nodes(tree.root_node, text, chunks, lang)
        return chunks or [text]
    
    def _detect_language(self, text: str) -> str:
        """Detección de lenguaje basada en patrones de seguridad"""
        patterns = {
            "python": r'\b(def |import |from |sys\.|os\.)',
            "c": r'#include|int main|printf\(',
            "cpp": r'#include|std::|using namespace',
            "asm": r'\b(mov |push |call |int 0x80|section)',
            "powershell": r'\$|Write-Host|Import-Module',
            "javascript": r'function |console\.|=>'
        }
        for lang, pattern in patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                return lang
        return "python"  # Default
    
    def _extract_nodes(self, node, text: str, chunks: List, lang: str, min_size: int = 50):
        """Extrae nodos técnicamente significativos"""
        # Definir nodos objetivo por lenguaje
        target_nodes = {
            "python": ["function_definition", "class_definition"],
            "c": ["function_definition", "struct_specifier"],
            "cpp": ["function_definition", "class_specifier"],
            "asm": ["function_definition", "instruction"],
            "powershell": ["function_definition", "cmdlet"],
            "javascript": ["function_declaration", "class_declaration"]
        }.get(lang, [])
        
        if node.type in target_nodes:
            chunk = text[node.start_byte:node.end_byte]
            if len(chunk) > min_size:
                chunks.append(chunk)
            return
        
        for child in node.children:
            self._extract_nodes(child, text, chunks, lang, min_size)

class LateChunkingDecorator(ChunkStrategy):
    """
    Optimización final de chunks basada en contexto semántico global.
    Fusiona fragmentos relacionados para mantener técnicas completas.
    """
    def __init__(self, wrapped: ChunkStrategy, context_model: str = "BAAI/bge-large-en-v1.5", 
                 device: str = "cuda", similarity_threshold: float = 0.82):
        self.wrapped = wrapped
        self.model_name = context_model
        self.device = device
        self.threshold = similarity_threshold
        self.tokenizer = AutoTokenizer.from_pretrained(context_model)
        self.model = AutoModel.from_pretrained(context_model).to(device)
    
    def chunk(self, text: str) -> List[str]:
        # 1. Generar chunks base
        base_chunks = self.wrapped.chunk(text)
        if len(base_chunks) <= 1:
            return base_chunks
        
        # 2. Embedding de contexto global (modo eficiente)
        global_embedding = self._embed_text(text[:2048])
        
        # 3. Fusionar chunks semánticamente relacionados
        merged_chunks = []
        current = base_chunks[0]
        
        for next_chunk in base_chunks[1:]:
            chunk_embedding = self._embed_text(next_chunk[:512])
            similarity = np.dot(global_embedding, chunk_embedding.T)[0][0]
            
            if similarity > self.threshold:
                current += "\n\n" + next_chunk
            else:
                merged_chunks.append(current)
                current = next_chunk
                
        merged_chunks.append(current)
        return merged_chunks
    
    def _embed_text(self, text: str) -> np.ndarray:
        """Embedding eficiente para fragmentos"""
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

class TechnicalChunkOptimizer(ChunkStrategy):
    """
    Optimizador final que aplica reglas específicas para contenido de seguridad:
    - Fusiona chunks pequeños relacionados
    - Asegura chunks autocontenidos
    - Añade contexto técnico
    """
    def __init__(self, wrapped: ChunkStrategy, min_size: int = 400):
        self.wrapped = wrapped
        self.min_size = min_size
    
    def chunk(self, text: str) -> List[str]:
        chunks = self.wrapped.chunk(text)
        optimized = []
        current = ""
        
        for chunk in chunks:
            # Fusionar chunks pequeños técnicamente relacionados
            if len(current) < self.min_size and self._are_related(current, chunk):
                current += "\n\n" + chunk if current else chunk
            else:
                if current: optimized.append(self._add_context(current))
                current = chunk
        
        if current: optimized.append(self._add_context(current))
        return optimized
    
    def _are_related(self, chunk1: str, chunk2: str) -> bool:
        """Determina si dos chunks tratan temas técnicos relacionados"""
        # Heurísticas simples para evitar dependencias de modelo
        keywords1 = set(re.findall(r'\b(ROP|ASLR|DEP|shellcode|exploit)\b', chunk1, re.IGNORECASE))
        keywords2 = set(re.findall(r'\b(ROP|ASLR|DEP|shellcode|exploit)\b', chunk2, re.IGNORECASE))
        return len(keywords1 & keywords2) > 0
    
    def _add_context(self, chunk: str) -> str:
        """Añade contexto técnico cuando es necesario"""
        if "shellcode" in chunk and "platform" not in chunk:
            return f"[Shellcode Technique]\n{chunk}"
        if "ROP" in chunk and not chunk.startswith("ROP Chain"):
            return f"ROP Technique: {chunk}"
        return chunk