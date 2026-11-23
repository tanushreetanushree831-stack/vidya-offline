"""Central configuration. Every value can be overridden with an environment variable."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Config:
    # --- models (Ollama) ---
    ollama_host: str | None      # None -> the ollama library default / OLLAMA_HOST
    chat_model: str
    embed_model: str
    num_ctx: int                 # context window handed to Ollama (bigger = more RAM)
    temperature: float
    # --- retrieval ---
    chunk_size: int              # characters per chunk
    chunk_overlap: int
    top_k: int
    max_distance: float          # cosine distance; hits above this are "not in the book"
    ocr_enabled: bool
    # --- paths ---
    data_dir: Path
    nllb_hf_dir: Path
    nllb_ct2_dir: Path

    @property
    def lancedb_dir(self) -> Path:
        return self.data_dir / "lancedb"

    @property
    def library_dir(self) -> Path:
        return self.data_dir / "library"


def load_config() -> Config:
    data_dir = Path(os.environ.get("VIDYA_DATA_DIR", ROOT / "data"))
    models_dir = Path(os.environ.get("VIDYA_MODELS_DIR", ROOT / "models"))
    return Config(
        ollama_host=os.environ.get("VIDYA_OLLAMA_HOST") or None,
        chat_model=os.environ.get("VIDYA_CHAT_MODEL", "llama3.2"),
        embed_model=os.environ.get("VIDYA_EMBED_MODEL", "nomic-embed-text"),
        num_ctx=_int("VIDYA_NUM_CTX", 4096),
        temperature=_float("VIDYA_TEMPERATURE", 0.3),
        chunk_size=_int("VIDYA_CHUNK_SIZE", 900),
        chunk_overlap=_int("VIDYA_CHUNK_OVERLAP", 150),
        top_k=_int("VIDYA_TOP_K", 4),
        max_distance=_float("VIDYA_MAX_DISTANCE", 0.65),
        ocr_enabled=os.environ.get("VIDYA_OCR", "1") != "0",
        data_dir=data_dir,
        nllb_hf_dir=models_dir / "nllb-200-distilled-600M",
        nllb_ct2_dir=models_dir / "nllb-200-distilled-600M-ct2",
    )
