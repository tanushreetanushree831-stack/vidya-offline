"""The student's bookshelf: a manifest plus the raw chunks (source of truth).

LanceDB is only an *index* built from these chunks, so changing the embedding model
simply means re-indexing - nothing has to be uploaded again.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class DocInfo:
    doc_id: str
    name: str
    pages: int
    chunks: int
    added: float = field(default_factory=time.time)
    indexed_with: list[str] = field(default_factory=list)   # embedding-model slugs


class Library:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.docs_dir = self.root / "docs"
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = self.root / "library.json"

    # ---- manifest ----
    def _read(self) -> dict[str, dict]:
        if not self.manifest.exists():
            return {}
        try:
            return json.loads(self.manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _write(self, data: dict[str, dict]) -> None:
        tmp = self.manifest.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self.manifest)

    def list_docs(self) -> list[DocInfo]:
        return sorted((DocInfo(**d) for d in self._read().values()), key=lambda d: d.added)

    def get(self, doc_id: str) -> DocInfo | None:
        d = self._read().get(doc_id)
        return DocInfo(**d) if d else None

    # ---- mutations ----
    def add_doc(self, info: DocInfo, chunks: list[dict]) -> None:
        with (self.docs_dir / f"{info.doc_id}.jsonl").open("w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        data = self._read()
        data[info.doc_id] = asdict(info)
        self._write(data)

    def mark_indexed(self, doc_id: str, embed_slug: str) -> None:
        data = self._read()
        if doc_id in data and embed_slug not in data[doc_id]["indexed_with"]:
            data[doc_id]["indexed_with"].append(embed_slug)
            self._write(data)

    def delete(self, doc_id: str) -> None:
        data = self._read()
        data.pop(doc_id, None)
        self._write(data)
        p = self.docs_dir / f"{doc_id}.jsonl"
        if p.exists():
            p.unlink()

    def load_chunks(self, doc_id: str) -> list[dict]:
        p = self.docs_dir / f"{doc_id}.jsonl"
        if not p.exists():
            return []
        with p.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
