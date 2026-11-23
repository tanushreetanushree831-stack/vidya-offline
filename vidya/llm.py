"""Thin, defensive wrapper around the Ollama Python client."""
from __future__ import annotations

import re
from typing import Any, Iterator, Sequence, Type, TypeVar

import ollama
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

EMBED_HINTS = ("embed", "bge", "minilm", "e5", "gte", "arctic")


class LLMError(RuntimeError):
    """Raised for any problem talking to the local model."""


def is_embed_model(name: str) -> bool:
    n = name.lower()
    return any(h in n for h in EMBED_HINTS)


def pick_default(models: Sequence[str], wanted: str) -> int:
    """Index of `wanted` inside `models`, tolerating ':latest' and tag differences."""
    if not models:
        return 0
    for i, m in enumerate(models):
        if m == wanted or m == f"{wanted}:latest":
            return i
    base = wanted.split(":")[0]
    for i, m in enumerate(models):
        if m.split(":")[0] == base:
            return i
    return 0


def _g(obj: Any, key: str, default: Any = None) -> Any:
    """Read a field from either an ollama response object or a plain dict."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


# --------------------------------------------------------------------------
# <think> ... </think> handling (DeepSeek-R1 and other reasoning models)
# --------------------------------------------------------------------------
_OPEN, _CLOSE = "<think>", "</think>"


def strip_think(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"<think>.*", "", text, flags=re.S)  # unterminated block
    return text.replace(_CLOSE, "").strip()


class ThinkStripper:
    """Streaming filter that removes <think>...</think> even when tags are split across chunks."""

    def __init__(self) -> None:
        self.buf = ""
        self.inside = False

    @staticmethod
    def _partial(buf: str, tag: str) -> int:
        for k in range(min(len(tag) - 1, len(buf)), 0, -1):
            if tag.startswith(buf[-k:]):
                return k
        return 0

    def feed(self, s: str) -> str:
        self.buf += s
        out: list[str] = []
        while True:
            if self.inside:
                i = self.buf.find(_CLOSE)
                if i == -1:
                    self.buf = self.buf[-(len(_CLOSE) - 1):]
                    break
                self.buf = self.buf[i + len(_CLOSE):]
                self.inside = False
            else:
                i = self.buf.find(_OPEN)
                if i == -1:
                    keep = self._partial(self.buf, _OPEN)
                    cut = len(self.buf) - keep
                    out.append(self.buf[:cut])
                    self.buf = self.buf[cut:]
                    break
                out.append(self.buf[:i])
                self.buf = self.buf[i + len(_OPEN):]
                self.inside = True
        return "".join(out)

    def flush(self) -> str:
        out = "" if self.inside else self.buf
        self.buf = ""
        return out


# --------------------------------------------------------------------------
def parse_json_model(raw: str, model_cls: Type[T]) -> T:
    """Parse JSON from a small model, tolerating prose or code fences around it."""
    raw = strip_think(raw)
    try:
        return model_cls.model_validate_json(raw)
    except (ValidationError, ValueError):
        pass
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in model output")
    return model_cls.model_validate_json(raw[start:end + 1])


class LLM:
    def __init__(self, model: str, host: str | None = None, num_ctx: int = 4096,
                 temperature: float = 0.3) -> None:
        self.model = model
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.client = ollama.Client(host=host) if host else ollama.Client()

    # ---- server / model discovery ----
    def is_up(self) -> bool:
        try:
            self.client.list()
            return True
        except Exception:
            return False

    def list_models(self) -> list[str]:
        resp = self.client.list()
        names = []
        for m in _g(resp, "models", []) or []:
            n = _g(m, "model") or _g(m, "name")
            if n:
                names.append(n)
        return sorted(names)

    def _options(self, temperature: float | None) -> dict:
        return {"num_ctx": self.num_ctx,
                "temperature": self.temperature if temperature is None else temperature}

    # ---- generation ----
    def chat_stream(self, messages: Sequence[dict], temperature: float | None = None) -> Iterator[str]:
        stripper = ThinkStripper()
        try:
            stream = self.client.chat(model=self.model, messages=list(messages), stream=True,
                                      options=self._options(temperature))
            for chunk in stream:
                piece = _g(_g(chunk, "message"), "content") or ""
                out = stripper.feed(piece)
                if out:
                    yield out
            tail = stripper.flush()
            if tail:
                yield tail
        except ollama.ResponseError as e:
            raise LLMError(f"Ollama error: {e}") from e
        except Exception as e:  # connection refused, timeouts...
            raise LLMError(f"Could not reach Ollama ({e}). Is it running?") from e

    def chat(self, messages: Sequence[dict], temperature: float | None = None) -> str:
        return strip_think("".join(self.chat_stream(messages, temperature)))

    def structured(self, messages: Sequence[dict], model_cls: Type[T],
                   temperature: float = 0.2, retries: int = 2) -> T:
        """Ask for JSON that matches `model_cls`; retry (colder) if the model rambles."""
        schema: Any = model_cls.model_json_schema()
        last: Exception | None = None
        for attempt in range(retries + 1):
            try:
                try:
                    resp = self.client.chat(model=self.model, messages=list(messages), format=schema,
                                            options=self._options(temperature if attempt == 0 else 0.0))
                except ollama.ResponseError:
                    # Older Ollama servers only understand format="json"
                    resp = self.client.chat(model=self.model, messages=list(messages), format="json",
                                            options=self._options(temperature if attempt == 0 else 0.0))
                content = _g(_g(resp, "message"), "content") or ""
                return parse_json_model(content, model_cls)
            except (ValidationError, ValueError) as e:
                last = e
            except Exception as e:
                raise LLMError(f"Could not reach Ollama ({e}). Is it running?") from e
        raise LLMError(f"The model did not return valid JSON after {retries + 1} tries: {last}")
