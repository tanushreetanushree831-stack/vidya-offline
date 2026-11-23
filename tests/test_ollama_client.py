"""Runs the REAL `ollama` python client against a tiny fake Ollama HTTP server."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from pydantic import BaseModel

from vidya.embeddings import EmbeddingError, OllamaEmbedder
from vidya.llm import LLM, LLMError

MODEL = {"model": "llama3.2:latest", "name": "llama3.2:latest", "modified_at": "2025-01-01T00:00:00Z",
         "size": 1, "digest": "abc", "details": {"format": "gguf", "family": "llama"}}
EMB = {**MODEL, "model": "nomic-embed-text:latest", "name": "nomic-embed-text:latest"}
SEEN = []


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/tags":
            self._send({"models": [MODEL, EMB]})
        else:
            self._send({}, 404)

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        SEEN.append((self.path, req))
        if self.path == "/api/embed":
            if req["model"] == "missing":
                return self._send({"error": "model 'missing' not found"}, 404)
            return self._send({"model": req["model"], "embeddings": [[3.0, 4.0]] * len(req["input"])})
        if self.path == "/api/chat":
            if req.get("stream", True):
                parts = ["<think>reason", "ing</think>Hel", "lo ", "world"]
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.end_headers()
                for p in parts:
                    self.wfile.write((json.dumps({"model": req["model"], "created_at": "2025-01-01T00:00:00Z",
                                                  "message": {"role": "assistant", "content": p},
                                                  "done": False}) + "\n").encode())
                self.wfile.write((json.dumps({"model": req["model"], "created_at": "2025-01-01T00:00:00Z",
                                              "message": {"role": "assistant", "content": ""},
                                              "done": True, "done_reason": "stop"}) + "\n").encode())
                return
            return self._send({"model": req["model"], "created_at": "2025-01-01T00:00:00Z", "done": True,
                               "message": {"role": "assistant", "content": 'Here you go: {"a": 7}'}})
        self._send({}, 404)


@pytest.fixture(scope="module")
def host():
    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_discovery(host):
    llm = LLM("llama3.2", host=host)
    assert llm.is_up()
    assert llm.list_models() == ["llama3.2:latest", "nomic-embed-text:latest"]
    assert not LLM("x", host="http://127.0.0.1:1").is_up()


def test_streaming_strips_think_and_sends_options(host):
    SEEN.clear()
    llm = LLM("llama3.2", host=host, num_ctx=3072, temperature=0.5)
    out = "".join(llm.chat_stream([{"role": "user", "content": "hi"}]))
    assert out == "Hello world"
    _, req = SEEN[-1]
    assert req["options"]["num_ctx"] == 3072 and req["options"]["temperature"] == 0.5
    assert llm.chat([{"role": "user", "content": "hi"}]) == "Hello world"


def test_structured_json(host):
    class Out(BaseModel):
        a: int
    SEEN.clear()
    r = LLM("llama3.2", host=host).structured([{"role": "user", "content": "x"}], Out)
    assert r.a == 7
    assert SEEN[-1][1]["format"]["properties"]["a"]["type"] == "integer"  # JSON-schema was sent


def test_embeddings_normalised_with_prefixes(host):
    SEEN.clear()
    e = OllamaEmbedder("nomic-embed-text", host=host)
    v = e.embed_documents(["a", "b", "c"], batch_size=2)
    assert len(v) == 3 and v[0] == pytest.approx([0.6, 0.8])
    assert SEEN[0][1]["input"] == ["search_document: a", "search_document: b"]
    e.embed_query("q")
    assert SEEN[-1][1]["input"] == ["search_query: q"]


def test_helpful_errors(host):
    with pytest.raises(EmbeddingError, match="ollama pull missing"):
        OllamaEmbedder("missing", host=host).embed_query("x")
    with pytest.raises(LLMError):
        list(LLM("x", host="http://127.0.0.1:1").chat_stream([{"role": "user", "content": "hi"}]))
