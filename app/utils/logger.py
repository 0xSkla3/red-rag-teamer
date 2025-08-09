# File: app/utils/logger.py
import logging
import os
import sys
from typing import Optional, Union

# Mapa robusto para strings de nivel
_LEVELS = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "NOTSET": logging.NOTSET,
}

def _coerce_level(level: Optional[Union[int, str]]) -> int:
    """
    Convierte un nivel (int|str|None) a int.
    Si es None, toma de env LOG_LEVEL (default INFO).
    """
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO")
    if isinstance(level, str):
        return _LEVELS.get(level.upper(), logging.INFO)
    if isinstance(level, int):
        return level
    return logging.INFO


def setup_logger(name: str, level: Optional[Union[int, str]] = None) -> logging.Logger:
    """
    Configura y devuelve un logger con nombre y nivel especificados.
    - Lee LOG_LEVEL del entorno si no se pasa explícito.
    - Evita duplicar handlers.
    - Desactiva propagate para no duplicar con root/uvicorn.
    """
    lvl = _coerce_level(level)
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(lvl)

    # Si ya tiene handlers, solo ajustamos niveles y salimos
    if logger.handlers:
        for h in logger.handlers:
            h.setLevel(lvl)
        return logger

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(lvl)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
