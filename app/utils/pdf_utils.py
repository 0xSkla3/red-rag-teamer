from __future__ import annotations

import hashlib
import re
from typing import List, Dict, Optional, Iterable, Tuple

import psutil
import tracemalloc
import fitz  # PyMuPDF
import pdfplumber
from rich.progress import Progress, TaskID

from app.utils.logger import setup_logger
from app.config import settings
from app.utils.chunk_strategies import ChunkStrategy

logger = setup_logger(__name__, level=settings.LOG_LEVEL)

# -------------------------- helpers de normalización --------------------------

_FENCE_RE = re.compile(r"(^```[a-zA-Z0-9_+\-]*\s*$)(.*?)(^```$)", re.MULTILINE | re.DOTALL)
_HARD_HYPHEN_RE = re.compile(r"(\w)-\n(\w)")
_SOFT_HYPHEN_RE = re.compile("\u00AD")
_CTRL_RE = re.compile(r"[\u200B\u200C\u200D\uFEFF]")  # zero-widths


def _normalize_text_light(text: str) -> str:
    if not text:
        return text
    t = _SOFT_HYPHEN_RE.sub("", text)
    t = _HARD_HYPHEN_RE.sub(r"\1\2", t)
    t = _CTRL_RE.sub("", t)
    t = re.sub(r"[ \t]+", " ", t)
    return t


def _tail_carry(text: str, base_chars: int = 400) -> str:
    if not text:
        return ""

    t_eval = text[-max(base_chars * 3, 1000):]
    fence_count = t_eval.count("```")
    braces_open = t_eval.count("{") - t_eval.count("}")
    paren_open = t_eval.count("(") - t_eval.count(")")
    indent_lines = [ln for ln in t_eval.splitlines()[-8:] if ln.strip()]
    indent_ratio = sum(1 for ln in indent_lines if re.match(r"^\s{2,}|\t", ln)) / max(len(indent_lines), 1)

    carry = text[-base_chars:]
    if fence_count % 2 == 1 or braces_open > 0 or paren_open > 0 or indent_ratio > 0.35:
        carry = text[-min(len(text), base_chars * 2):]
    return carry


def _dedupe_key(s: str) -> str:
    return hashlib.md5(s.strip().encode("utf-8", errors="ignore")).hexdigest()


# --------------------------- extracción por PyMuPDF ---------------------------

def _extract_text_fitz(page: fitz.Page, mode: str = "blocks", sort: bool = True) -> str:
    """
    Extrae texto en modos livianos. Evitamos 'xhtml' (muy costoso).
    """
    try:
        if mode == "blocks":
            blocks = page.get_text("blocks", sort=sort)
            try:
                blocks = sorted(blocks, key=lambda b: (round(b[1], 2), round(b[0], 2)))
            except Exception:
                pass
            text = "\n".join(
                b[4] for b in blocks
                if isinstance(b, (list, tuple)) and len(b) >= 5 and b[4]
            )
            if text and text.strip():
                return text
        # Fallback rápido a "text"
        t = page.get_text("text", sort=sort)
        return t or ""
    except Exception as e:
        logger.debug(f"fitz.get_text fallback due to: {e}")
        try:
            return page.get_text("text") or ""
        except Exception:
            return ""


# --------------------------- tablas via pdfplumber ----------------------------

def _maybe_extract_tables(
    page_plumb: pdfplumber.page.Page,
    enable: bool,
    table_settings: Optional[dict] = None
) -> List[List[List[Optional[str]]]]:
    if not enable:
        return []
    try:
        # Heurística barata: si no hay líneas/rectángulos, probablemente no hay tabla “line-based”
        lines = len(page_plumb.lines or [])
        rects = len(page_plumb.rects or [])
        if (lines + rects) < 6:
            return []
        default_settings = {
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
            "intersection_tolerance": 3,
            "snap_tolerance": 3,
            "join_tolerance": 3,
            "edge_min_length": 3,
        }
        if table_settings:
            default_settings.update(table_settings)
        tables = page_plumb.extract_tables(table_settings=default_settings)
        return tables or []
    except Exception as e:
        logger.debug(f"pdfplumber.extract_tables skipped: {e}")
        return []


# -------------------------- iterador de páginas dual --------------------------

def _iter_pages_dual(
    pdf_path: str,
    enable_tables: bool = True,
    table_settings: Optional[dict] = None,
    fitz_mode: str = "blocks",
    fitz_sort: bool = True
) -> Iterable[Tuple[int, str, List[List[List[Optional[str]]]]]]:
    with fitz.open(pdf_path) as fdoc:
        if not enable_tables:
            for i, page in enumerate(fdoc, start=1):
                text = _extract_text_fitz(page, mode=fitz_mode, sort=fitz_sort)
                yield i, text, []
        else:
            # pdfplumber se abre una sola vez para todo el documento (lazy pages)
            with pdfplumber.open(pdf_path) as pdoc:
                total = len(fdoc)
                for i in range(total):
                    fpage = fdoc[i]
                    text = _extract_text_fitz(fpage, mode=fitz_mode, sort=fitz_sort)
                    ppage = pdoc.pages[i]
                    tables = _maybe_extract_tables(ppage, enable=True, table_settings=table_settings)
                    yield i + 1, text, tables


# ------------------------------ API principal --------------------------------

async def extract_chunks_from_pdf(
    pdf_path: str,
    strategy: ChunkStrategy,
    carry_chars: int = settings.PDF_CARRY_CHARS,
    trim_carry_on_first_chunk: bool = True,
    progress: Optional[Progress] = None,
    prefer_tables: bool = True,                # default TRUE
    table_settings: Optional[dict] = None,
    dedupe: bool = True,
) -> List[Dict]:
    """
    Extrae y chunkea el PDF página a página usando:
      • PyMuPDF (texto principal en *reading order*),
      • pdfplumber (opcional) solo para *tablas*.

    Retorna: [{'id': '0001-0001', 'text': '...', 'meta': {source,page,chunk,content_type}}]
    """
    process = psutil.Process()
    tracemalloc.start()
    initial_rss = process.memory_info().rss / 1e6
    logger.info(f"🔍 Start extracting `{pdf_path}` | initial RSS: [bold]{initial_rss:.1f} MB[/]")

    docs: List[Dict] = []
    prev_tail: str = ""
    seen_hashes: set[str] = set() if dedupe else set()

    p_task: Optional[TaskID] = None

    try:
        iterator = _iter_pages_dual(
            pdf_path,
            enable_tables=prefer_tables,
            table_settings=table_settings,
            fitz_mode="blocks",
            fitz_sort=True
        )

        total_pages = None
        try:
            with fitz.open(pdf_path) as tmp:
                total_pages = len(tmp)
        except Exception:
            pass

        if progress and total_pages:
            p_task = progress.add_task("Páginas PDF", total=total_pages)

        for page_no, raw_text, tables in iterator:
            rss_before = process.memory_info().rss / 1e6
            logger.debug(f"📄 Page {page_no}{'/{total_pages}' if total_pages else ''} | RSS before: {rss_before:.1f} MB")

            page_text = _normalize_text_light(raw_text or "")
            working_text = (prev_tail + "\n" + page_text) if prev_tail else page_text
            base_carry_len = len(prev_tail)

            # Guard: si la página no tiene texto útil, avanzamos progreso y seguimos.
            if not working_text or not working_text.strip():
                logger.debug(f"⚠️  Page {page_no}: empty working_text (len={len(working_text) if working_text else 0})")
                prev_tail = ""
                if progress and p_task is not None:
                    progress.advance(p_task)
                continue

            local_chunks = strategy.chunk(working_text)
            logger.debug(f"🧩 Page {page_no}: text_len={len(working_text)} | local_chunks={len(local_chunks)}")

            for idx, ch in enumerate(local_chunks, start=1):
                ch_out = ch
                if trim_carry_on_first_chunk and base_carry_len > 0 and idx == 1:
                    prefix = prev_tail[-base_carry_len:]
                    if ch_out.startswith(prefix):
                        ch_out = ch_out[base_carry_len:]

                ch_out = (ch_out or "").strip()
                if not ch_out:
                    continue

                if dedupe:
                    h = _dedupe_key(ch_out[:2048])
                    if h in seen_hashes:
                        continue
                    if len(seen_hashes) > 100_000:
                        seen_hashes.clear()
                    seen_hashes.add(h)

                docs.append({
                    'id': f"{page_no:04d}-{idx:04d}",
                    'text': ch_out,
                    'meta': {
                        'source': pdf_path,
                        'page': page_no,
                        'chunk': idx,
                        'content_type': 'text'
                    }
                })

            if tables:
                for t_idx, table in enumerate(tables, start=1):
                    rows = [",".join((cell or "").strip() for cell in row) for row in table or []]
                    table_text = "[TABLE]\n" + "\n".join(rows)
                    if table_text.strip():
                        if dedupe:
                            h = _dedupe_key(table_text[:2048])
                            if h in seen_hashes:
                                continue
                            seen_hashes.add(h)
                        docs.append({
                            'id': f"{page_no:04d}-T{t_idx:03d}",
                            'text': table_text,
                            'meta': {
                                'source': pdf_path,
                                'page': page_no,
                                'chunk': t_idx,
                                'content_type': 'table'
                            }
                        })

            prev_tail = _tail_carry(page_text, carry_chars)

            current, peak = tracemalloc.get_traced_memory()
            rss_after = process.memory_info().rss / 1e6
            logger.debug(
                f"✅ Finished page {page_no} | page_text_len={len(page_text)} | RSS after: {rss_after:.1f} MB | "
                f"tracemalloc current: {current/1e6:.1f} MB | peak: {peak/1e6:.1f} MB"
            )
            tracemalloc.clear_traces()

            if progress and p_task is not None:
                progress.advance(p_task)

    finally:
        tracemalloc.stop()
        final_rss = process.memory_info().rss / 1e6
        logger.info(f"🏁 Extraction completed | total chunks: [bold]{len(docs)}[/] | final RSS: [bold]{final_rss:.1f} MB[/]")

    return docs
