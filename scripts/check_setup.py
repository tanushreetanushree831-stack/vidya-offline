"""Run this first if anything misbehaves:  python scripts\\check_setup.py"""
import importlib
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OK, BAD, WARN = "[ OK ]", "[FAIL]", "[WARN]"
problems = 0


def say(tag, msg):
    global problems
    problems += tag == BAD
    print(f"{tag} {msg}")


print("=== Vidya Offline - setup check ===\n")

v = sys.version_info
if (3, 10) <= v[:2] <= (3, 12):
    say(OK, f"Python {v.major}.{v.minor}.{v.micro}")
else:
    say(WARN, f"Python {v.major}.{v.minor} - the project is tested on 3.10-3.12; 3.11 or 3.12 is safest.")

for mod, pip in [("streamlit", "streamlit"), ("ollama", "ollama"), ("lancedb", "lancedb"), ("pydantic", "pydantic"),
                 ("pypdf", "pypdf"), ("pypdfium2", "pypdfium2"), ("docx", "python-docx"), ("pptx", "python-pptx"),
                 ("rapidocr_onnxruntime", "rapidocr-onnxruntime"), ("PIL", "Pillow"), ("numpy", "numpy"),
                 ("psutil", "psutil")]:
    try:
        m = importlib.import_module(mod)
        say(OK, f"{pip} {getattr(m, '__version__', '')}")
    except Exception as e:
        say(BAD, f"{pip} not importable ({e.__class__.__name__}). Fix: pip install -r requirements.txt")

# ---- hardware ----
try:
    import psutil
    gb = psutil.virtual_memory().total / 1024 ** 3
    free = shutil.disk_usage(Path(__file__).resolve().parent.parent).free / 1024 ** 3
    rec = "llama3.2:1b  (also try qwen2.5:1.5b)" if gb < 6 else "llama3.2 (3B)"
    say(OK, f"RAM {gb:.1f} GB, free disk {free:.1f} GB  ->  suggested chat model: {rec}")
    if free < 6:
        say(WARN, "Less than 6 GB free disk: models need roughly 2-5 GB.")
except Exception:
    pass

# ---- Ollama ----
from vidya.config import load_config   # noqa: E402
from vidya.llm import LLM, is_embed_model  # noqa: E402
from vidya.translate import Translator   # noqa: E402

cfg = load_config()
llm = LLM(cfg.chat_model, host=cfg.ollama_host)
if llm.is_up():
    say(OK, "Ollama server is running")
    models = llm.list_models()
    chat = [m for m in models if not is_embed_model(m)]
    emb = [m for m in models if is_embed_model(m)]
    say(OK if chat else BAD, f"chat models: {', '.join(chat) or 'none  -> ollama pull llama3.2'}")
    say(OK if emb else BAD, f"embedding models: {', '.join(emb) or 'none  -> ollama pull nomic-embed-text'}")
    if emb:
        try:
            from vidya.embeddings import OllamaEmbedder
            vec = OllamaEmbedder(emb[0], cfg.ollama_host).embed_query("test")
            say(OK, f"embedding test passed (dimension {len(vec)})")
        except Exception as e:
            say(BAD, f"embedding test failed: {e}")
    if chat:
        try:
            out = LLM(chat[0], cfg.ollama_host).chat([{"role": "user", "content": "Reply with the single word: ready"}])
            say(OK, f"chat test passed ({chat[0]} said: {out.strip()[:40]!r})")
        except Exception as e:
            say(BAD, f"chat test failed: {e}")
else:
    say(BAD, "Ollama is not running. Install from https://ollama.com/download and start it.")

# ---- translation ----
t = Translator(cfg.nllb_ct2_dir, cfg.nllb_hf_dir, llm=llm if llm.is_up() else None)
b = t.backend_name()
say(OK if b in ("ct2", "hf") else WARN, f"translation: {t.describe()}")
if b not in ("ct2", "hf"):
    print("       For good Hindi/Kannada/etc. install it:  pip install -r requirements-translation.txt")
    print("                                               python scripts\\download_translation_model.py --slim")

print("\n" + ("Everything needed is ready." if not problems else f"{problems} problem(s) found - fix the [FAIL] lines above."))
sys.exit(1 if problems else 0)
