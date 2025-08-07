# File: app/clients/embedding_client.py
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from app.utils.logger import setup_logger
from typing import List, Union

logger = setup_logger(__name__)

class EmbeddingClient:
    def __init__(self, model_name: str, device: str, use_nli: bool = True):
        self.device = device
        self.use_nli = use_nli
        
        if use_nli:
            # Modelos NLI optimizados para inferencia local
            nli_models = {
                "nli-mpnet": "sentence-transformers/all-mpnet-base-v2",
                "nli-minilm": "sentence-transformers/all-MiniLM-L6-v2",
                "code-nli": "codebert-nli"
            }
            model_path = nli_models.get(model_name, "sentence-transformers/all-mpnet-base-v2")
            
            self.model = SentenceTransformer(model_path, device=device)
            logger.info(f"Loaded NLI-optimized model '{model_path}' on '{device}'")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(device)
            logger.info(f"Loaded standard model '{model_name}' on '{device}'")

    def embed(self, text: Union[str, List[str]]) -> List[List[float]]:
        """Genera embeddings para texto individual o batch"""
        if isinstance(text, str):
            text = [text]
            
        logger.debug(f"Embedding {len(text)} chunks (avg len: {sum(len(t) for t in text)/len(text):.0f} chars)")
        
        if self.use_nli:
            # Método optimizado para modelos NLI
            embeddings = self.model.encode(
                text,
                batch_size=32,
                convert_to_tensor=True,
                device=self.device
            )
            return embeddings.cpu().tolist()
        else:
            # Método estándar para otros modelos
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Pooling inteligente: mean pooling con máscara de atención
            attention_mask = inputs['attention_mask']
            last_hidden = outputs.last_hidden_state
            embeddings = (last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1).unsqueeze(-1)
            return embeddings.cpu().tolist()