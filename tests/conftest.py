import hashlib
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class FakeEmbedder:
    """Deterministic bag-of-words embedder so retrieval can be tested without Ollama."""
    model = "fake-embed"
    DIM = 256

    def _vec(self, text):
        v = [0.0] * self.DIM
        for w in re.findall(r"\w+", text.lower()):
            h = int(hashlib.md5(w.encode()).hexdigest(), 16) % self.DIM
            v[h] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts, batch_size=16, progress=None):
        out = [self._vec(t) for t in texts]
        if progress:
            progress(1.0)
        return out

    def embed_query(self, text):
        return self._vec(text)


class FakeLLM:
    """Returns canned output; records the messages it was given."""
    def __init__(self, reply="OK", structured=None):
        self.reply, self._structured, self.calls = reply, structured, []

    def chat(self, messages, temperature=None):
        self.calls.append(messages)
        return self.reply

    def chat_stream(self, messages, temperature=None):
        self.calls.append(messages)
        yield from (w + " " for w in self.reply.split())

    def structured(self, messages, model_cls, temperature=0.2, retries=2):
        self.calls.append(messages)
        return self._structured
