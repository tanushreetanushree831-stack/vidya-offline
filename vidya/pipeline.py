"""Glue: file bytes -> pages -> chunks -> library -> embeddings -> vector index."""
from __future__ import annotations

import hashlib
from typing import Callable

from .config import Config
from .embeddings import OllamaEmbedder
from .ingest import chunk_pages, extract_pages
from .library import DocInfo, Library
from .store import VectorStore

Progress = Callable[[float, str], None]


def make_doc_id(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()[:12]


def index_document(library: Library, store: VectorStore, embedder: OllamaEmbedder,
                   doc_id: str, progress: Progress | None = None) -> int:
    """(Re)build the vector index for one stored document. Returns number of chunks indexed."""
    chunks = library.load_chunks(doc_id)
    if not chunks:
        return 0
    store.delete_doc(doc_id)  # makes re-indexing idempotent
    vectors = embedder.embed_documents(
        [c["text"] for c in chunks],
        progress=(lambda f: progress(f, "Embedding passages")) if progress else None)
    rows = [{**c, "vector": v} for c, v in zip(chunks, vectors)]
    store.add(rows)
    library.mark_indexed(doc_id, store.embed_slug)
    return len(rows)


def ingest_file(name: str, data: bytes, cfg: Config, library: Library, store: VectorStore,
                embedder: OllamaEmbedder, progress: Progress | None = None
                ) -> tuple[DocInfo | None, list[str], bool]:
    """Returns (doc, warnings, already_present)."""
    doc_id = make_doc_id(data)
    existing = library.get(doc_id)
    if existing:
        if store.embed_slug not in existing.indexed_with:
            index_document(library, store, embedder, doc_id, progress)
            existing = library.get(doc_id)
        return existing, [], True

    if progress:
        progress(0.0, "Reading file")
    pages, warnings = extract_pages(
        name, data, ocr=cfg.ocr_enabled,
        progress=(lambda f: progress(f, "Reading pages")) if progress else None)
    chunks = chunk_pages(pages, name, doc_id, cfg.chunk_size, cfg.chunk_overlap)
    if not chunks:
        return None, warnings + ["Nothing readable was found in this file."], False

    info = DocInfo(doc_id=doc_id, name=name,
                   pages=max(p.number for p in pages), chunks=len(chunks))
    library.add_doc(info, chunks)
    index_document(library, store, embedder, doc_id, progress)
    return library.get(doc_id), warnings, False
