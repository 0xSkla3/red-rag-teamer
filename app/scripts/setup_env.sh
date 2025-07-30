#!/usr/bin/env bash
# setup_env.sh — Script para crear y configurar el entorno virtual del RAG service

#set -euo pipefail

# 1) Crear el virtualenv en .venv
python3 -m venv .venv

# 2) Activar el entorno
#    En Linux/macOS:
source .venv/bin/activate

# 3) Actualizar pip
pip install --upgrade pip

# 4) Instalar dependencias
pip install --no-cache-dir -r requirements.txt

echo "-> Entorno virtual creado y dependencias instaladas."
