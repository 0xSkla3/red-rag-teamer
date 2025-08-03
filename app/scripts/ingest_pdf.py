# File: scripts/ingest_pdf.py
import asyncio
import sys

from app.utils.pdf_utils import extract_chunks_from_pdf
from app.services.index_service import IndexService

async def main(pdf_path: str):
    # 1) Extrae y chunkéa el PDF
    print(f"▶ Extrayendo chunks de {pdf_path}...")
    docs = await extract_chunks_from_pdf(pdf_path)
    print(f"   → Generados {len(docs)} chunks.")

    # 2) Indexa en Qdrant
    indexer = IndexService()
    print("▶ Indexando en Qdrant...")
    await indexer.index_documents(docs)
    print("   ✓ Indexación completada.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python ingest_pdf.py ruta/al/documento.pdf")
        sys.exit(1)
    pdf_path = sys.argv[1]
    asyncio.run(main(pdf_path))