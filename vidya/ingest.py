"""Stage 1 & 2: read PDFs / DOCX / PPTX / images / text, OCR when needed, split into chunks."""
from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Callable

SUPPORTED = {".pdf", ".docx", ".pptx", ".txt", ".md", ".png", ".jpg", ".jpeg", ".bmp", ".webp"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


class IngestError(RuntimeError):
    pass


@dataclass
class Page:
    number: int
    text: str


# ----------------------------------------------------------------------------- OCR
_ocr_engine = None
_ocr_failed = False


def _get_ocr():
    """Lazy PP-OCR (ONNX) engine. Models ship inside the pip package, so it works offline."""
    global _ocr_engine, _ocr_failed
    if _ocr_engine is None and not _ocr_failed:
        try:
            from rapidocr_onnxruntime import RapidOCR
            _ocr_engine = RapidOCR()
        except Exception:
            _ocr_failed = True
    return _ocr_engine


def ocr_image(pil_image) -> str:
    engine = _get_ocr()
    if engine is None:
        return ""
    import numpy as np
    result, _ = engine(np.array(pil_image.convert("RGB")))
    if not result:
        return ""
    # result rows: [box, text, score]; sort roughly top-to-bottom, left-to-right
    rows = sorted(result, key=lambda r: (round(r[0][0][1] / 12), r[0][0][0]))
    lines, last_y = [], None
    for box, text, _score in rows:
        y = round(box[0][1] / 12)
        if last_y is not None and y == last_y:
            lines[-1] += " " + text
        else:
            lines.append(text)
        last_y = y
    return "\n".join(lines)


# ----------------------------------------------------------------------------- readers
def _read_pdf(data: bytes, ocr: bool, warnings: list[str],
              progress: Callable[[float], None] | None) -> list[Page]:
    from pypdf import PdfReader
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            reader.decrypt("")
        n = len(reader.pages)
    except Exception as e:
        raise IngestError(f"Could not open this PDF ({e}).") from e

    pdfium_doc = None
    pages: list[Page] = []
    scanned = 0
    for i in range(n):
        try:
            text = reader.pages[i].extract_text() or ""
        except Exception:
            text = ""
        if len(text.strip()) < 25 and ocr:
            try:
                if pdfium_doc is None:
                    import pypdfium2 as pdfium
                    pdfium_doc = pdfium.PdfDocument(data)
                img = pdfium_doc[i].render(scale=2.0).to_pil()
                text = ocr_image(img)
                scanned += 1
            except Exception:
                pass
        if text.strip():
            pages.append(Page(i + 1, text))
        if progress:
            progress((i + 1) / n)
    if scanned:
        warnings.append(f"{scanned} page(s) had no text layer and were read with OCR.")
    if not pages:
        warnings.append("No text could be extracted. If this is a scan, make sure OCR is enabled.")
    return pages


def _read_docx(data: bytes) -> list[Page]:
    import docx
    d = docx.Document(io.BytesIO(data))
    parts = [p.text for p in d.paragraphs if p.text.strip()]
    for table in d.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    # Word has no fixed pages; group ~3000 characters as a pseudo-page.
    pages, buf, n = [], "", 1
    for p in parts:
        buf += p + "\n\n"
        if len(buf) >= 3000:
            pages.append(Page(n, buf))
            buf, n = "", n + 1
    if buf.strip():
        pages.append(Page(n, buf))
    return pages


def _read_pptx(data: bytes, ocr: bool) -> list[Page]:
    from pptx import Presentation
    prs = Presentation(io.BytesIO(data))
    pages = []
    for i, slide in enumerate(prs.slides, start=1):
        parts = []
        for shape in slide.shapes:
            if shape.has_text_frame and shape.text_frame.text.strip():
                parts.append(shape.text_frame.text)
            if getattr(shape, "has_table", False) and shape.has_table:
                for row in shape.table.rows:
                    cells = [c.text.strip() for c in row.cells if c.text.strip()]
                    if cells:
                        parts.append(" | ".join(cells))
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame.text.strip():
            parts.append(slide.notes_slide.notes_text_frame.text)
        if parts:
            pages.append(Page(i, "\n\n".join(parts)))
    return pages


def _read_image(data: bytes, ocr: bool, warnings: list[str]) -> list[Page]:
    if not ocr:
        warnings.append("OCR is disabled, so image files cannot be read.")
        return []
    from PIL import Image
    text = ocr_image(Image.open(io.BytesIO(data)))
    if not text.strip():
        warnings.append("OCR found no readable text in this image.")
        return []
    return [Page(1, text)]


def extract_pages(name: str, data: bytes, ocr: bool = True,
                  progress: Callable[[float], None] | None = None) -> tuple[list[Page], list[str]]:
    """Return (pages, warnings) for any supported file."""
    ext = "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""
    warnings: list[str] = []
    if ext not in SUPPORTED:
        raise IngestError(f"'{ext or name}' files are not supported. Use: {', '.join(sorted(SUPPORTED))}")
    if ext == ".pdf":
        pages = _read_pdf(data, ocr, warnings, progress)
    elif ext == ".docx":
        pages = _read_docx(data)
    elif ext == ".pptx":
        pages = _read_pptx(data, ocr)
    elif ext in IMAGE_EXT:
        pages = _read_image(data, ocr, warnings)
    else:
        pages = [Page(1, data.decode("utf-8", errors="ignore"))]
    return pages, warnings


# ----------------------------------------------------------------------------- chunking
_SENT_SPLIT = re.compile(r"(?<=[.!?\u0964\u0965])\s+")


def clean_text(text: str) -> str:
    text = text.replace("\r", "")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)          # re-join hyphenated line breaks
    paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = [re.sub(r"\s*\n\s*", " ", p) for p in paragraphs]
    paragraphs = [re.sub(r"[ \t\u00a0]+", " ", p).strip() for p in paragraphs]
    return "\n\n".join(p for p in paragraphs if p)


def split_sentences(text: str) -> list[str]:
    out: list[str] = []
    for para in clean_text(text).split("\n\n"):
        out.extend(s.strip() for s in _SENT_SPLIT.split(para) if s.strip())
    return out


def _hard_split(sentence: str, size: int) -> list[str]:
    return [sentence[i:i + size] for i in range(0, len(sentence), size)]


def chunk_pages(pages: list[Page], source: str, doc_id: str,
                size: int = 900, overlap: int = 150, min_chars: int = 30) -> list[dict]:
    """Sentence-aware chunks that remember their page number (used for citations)."""
    chunks: list[dict] = []
    for page in pages:
        sentences: list[str] = []
        for s in split_sentences(page.text):
            sentences.extend(_hard_split(s, size) if len(s) > size else [s])

        buf: list[str] = []
        length = 0

        def flush() -> None:
            nonlocal buf, length
            text = " ".join(buf).strip()
            if len(text) >= min_chars:
                chunks.append({
                    "chunk_id": f"{doc_id}-{len(chunks):05d}",
                    "doc_id": doc_id, "source": source, "page": page.number, "text": text,
                })
            # carry the tail of this chunk into the next one as overlap
            carry: list[str] = []
            carried = 0
            for s in reversed(buf):
                if carried + len(s) > overlap:
                    break
                carry.insert(0, s)
                carried += len(s) + 1
            buf, length = carry, carried

        for s in sentences:
            if length + len(s) + 1 > size and buf:
                flush()
            buf.append(s)
            length += len(s) + 1
        if buf and (len(" ".join(buf)) >= min_chars):
            text = " ".join(buf).strip()
            # avoid emitting a chunk that is only overlap already present in the previous one
            if not chunks or chunks[-1]["page"] != page.number or text not in chunks[-1]["text"]:
                chunks.append({
                    "chunk_id": f"{doc_id}-{len(chunks):05d}",
                    "doc_id": doc_id, "source": source, "page": page.number, "text": text,
                })
    return chunks
