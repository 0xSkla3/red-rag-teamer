# app/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    OLLAMA_URL: str = "http://ollama:11434"
    QDRANT_URL: str = "http://qdrant:6333"
    MODEL_NAME: str = "deepseek-r1:14b"
    HOST: str = "0.0.0.0"
    PORT: int = 5000
    TOP_K: int = 5
    EMBED_DIM: int = 768

    class Config:
        env_file = ".env"

settings = Settings()