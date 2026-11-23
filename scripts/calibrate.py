"""Tune the 'is this question in my book?' threshold.

Usage (Ollama must be running, at least one book indexed):
    python scripts/calibrate.py "What is photosynthesis?" "Who won the 2011 cricket world cup?"
    python scripts/calibrate.py            # uses a few built-in off-topic questions only

It prints the cosine distance of the closest passage for each question. Pick a limit that is
LARGER than the distances of questions that should match and SMALLER than those that should not,
then set it with the 'Book-match strictness' slider (or VIDYA_MAX_DISTANCE).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vidya.config import load_config          # noqa: E402
from vidya.embeddings import OllamaEmbedder   # noqa: E402
from vidya.library import Library             # noqa: E402
from vidya.store import VectorStore           # noqa: E402

OFF_TOPIC = ["Who won the cricket world cup in 2011?", "What is the capital of Australia?",
             "Write a song about friendship", "How do I cook biryani?", "Who is the Prime Minister of Japan?"]


def main() -> None:
    cfg = load_config()
    lib = Library(cfg.library_dir)
    if not lib.list_docs():
        sys.exit("No books indexed yet. Add one in the app first.")
    store = VectorStore(cfg.lancedb_dir, cfg.embed_model)
    emb = OllamaEmbedder(cfg.embed_model, cfg.ollama_host)
    mine = sys.argv[1:]
    print(f"Embedding model: {cfg.embed_model}   current limit: {cfg.max_distance}\n")
    for label, qs in (("YOUR QUESTIONS", mine), ("OFF-TOPIC SAMPLES (should be > limit)", OFF_TOPIC)):
        if not qs:
            continue
        print(label)
        for q in qs:
            hits = store.search(emb.embed_query(q), k=1)
            d = hits[0]["_distance"] if hits else float("nan")
            where = f"{hits[0]['source']} p.{hits[0]['page']}" if hits else "-"
            print(f"  {d:5.2f}  {q}   -> {where}")
        print()


if __name__ == "__main__":
    main()
