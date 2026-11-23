"""Embeddings through Ollama (no PyTorch needed)."""
from __future__ import annotations

import math
from typing import Callable, Sequence

import ollama

from .llm import _g

# (query_prefix, document_prefix) required by some embedding models.
_PREFIXES = {
    "nomic-embed-text": ("search_query: ", "search_document: "),
}


class EmbeddingError(RuntimeError):
    pass


def prefixes_for(model: str) -> tuple[str, str]:
    base = model.split(":")[0]
    return _PREFIXES.get(base, ("", ""))


def _normalize(v: Sequence[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


class OllamaEmbedder:
    def __init__(self, model: str, host: str | None = None) -> None:
        self.model = model
        self.client = ollama.Client(host=host) if host else ollama.Client()
        self.q_prefix, self.d_prefix = prefixes_for(model)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        try:
            resp = self.client.embed(model=self.model, input=texts)
        except ollama.ResponseError as e:
            raise EmbeddingError(
                f"Embedding model '{self.model}' failed: {e}. "
                f"Install it with:  ollama pull {self.model}") from e
        except Exception as e:
            raise EmbeddingError(f"Could not reach Ollama ({e}). Is it running?") from e
        vecs = _g(resp, "embeddings")
        if not vecs or len(vecs) != len(texts):
            raise EmbeddingError("Ollama returned an unexpected embedding response.")
        return [_normalize(v) for v in vecs]

    def embed_documents(self, texts: Sequence[str], batch_size: int = 16,
                        progress: Callable[[float], None] | None = None) -> list[list[float]]:
        out: list[list[float]] = []
        total = len(texts)
        for i in range(0, total, batch_size):
            batch = [self.d_prefix + t for t in texts[i:i + batch_size]]
            out.extend(self._embed(batch))
            if progress:
                progress(min(1.0, (i + len(batch)) / max(total, 1)))
        return out

    def embed_query(self, text: str) -> list[float]:
        return self._embed([self.q_prefix + text])[0]
