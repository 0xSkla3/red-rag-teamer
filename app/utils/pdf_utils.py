# File: app/utils/pdf_utils.py
import pdfplumber

# Extrae y chunkea texto del PDF
def chunk_text(text: str, max_chars: int = 2000, overlap: int = 200) -> list[str]:
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + max_chars, length)
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

async def extract_chunks_from_pdf(pdf_path: str) -> list[dict]:
    with pdfplumber.open(pdf_path) as pdf:
        full_text = "\n".join(page.extract_text() or '' for page in pdf.pages)
    chunks = chunk_text(full_text)
    return [
        { 'id': f"{pdf_path}-{i}", 'text': chunk, 'meta': { 'source': pdf_path, 'chunk': i } }
        for i, chunk in enumerate(chunks)
    ]