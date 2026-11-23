"""Embedded, serverless vector store (LanceDB). One table per embedding model."""
from __future__ import annotations

import re
from pathlib import Path

import lancedb


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _q(value: str) -> str:
    return value.replace("'", "''")


class VectorStore:
    def __init__(self, path: str | Path, embed_model: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(path))
        self.embed_slug = slug(embed_model)
        self.table_name = f"chunks_{self.embed_slug}"

    def _open(self):
        try:
            return self.db.open_table(self.table_name)
        except Exception:
            return None

    def add(self, rows: list[dict]) -> None:
        """rows: dicts with chunk_id, doc_id, source, page, text, vector."""
        if not rows:
            return
        tbl = self._open()
        if tbl is None:
            self.db.create_table(self.table_name, data=rows)
        else:
            tbl.add(rows)

    def delete_doc(self, doc_id: str) -> None:
        tbl = self._open()
        if tbl is not None:
            tbl.delete(f"doc_id = '{_q(doc_id)}'")

    def count(self) -> int:
        tbl = self._open()
        return tbl.count_rows() if tbl is not None else 0

    def search(self, vector: list[float], k: int = 4, doc_ids: list[str] | None = None) -> list[dict]:
        """Nearest chunks by cosine distance (0 = identical). Each hit has a `_distance` key."""
        tbl = self._open()
        if tbl is None:
            return []
        q = tbl.search(vector).metric("cosine").limit(k)
        if doc_ids:
            q = q.where("doc_id IN (" + ", ".join(f"'{_q(d)}'" for d in doc_ids) + ")")
        hits = q.to_list()
        for h in hits:
            h.pop("vector", None)
        return hits
