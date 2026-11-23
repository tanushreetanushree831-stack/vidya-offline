"""Rule-based extractive summariser (TextRank over TF-IDF similarity).

Pure Python, no model, ~0 MB RAM: this is the "graph engine" box of the architecture.
It is used (a) as an instant summary for very slow PCs and (b) to compress long
material so it fits a small model's context window.
"""
from __future__ import annotations

import math
import re
from collections import Counter

from .ingest import split_sentences

_STOP = set("""a an and are as at be been but by can could did do does for from had has have he her his how i if in
into is it its may might more most no not of on or our she so some such than that the their them then there these they
this those to too was we were what when where which who whom why will with would you your also about after all any
because both each few other over own same should very""".split())

_WORD = re.compile(r"[^\W\d_]{3,}", re.UNICODE)


def _tokens(sentence: str) -> list[str]:
    return [w for w in (m.lower() for m in _WORD.findall(sentence)) if w not in _STOP]


def dedupe(sentences: list[str]) -> list[str]:
    seen, out = set(), []
    for s in sentences:
        key = s.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(s.strip())
    return out


def rank_sentences(sentences: list[str], max_nodes: int = 400, damping: float = 0.85,
                   iterations: int = 30) -> list[float]:
    n = len(sentences)
    if n == 0:
        return []
    if n == 1:
        return [1.0]
    if n > max_nodes:  # keep the O(n^2) graph small on weak CPUs
        step = n / max_nodes
        picked = [int(i * step) for i in range(max_nodes)]
        sub = rank_sentences([sentences[i] for i in picked], max_nodes, damping, iterations)
        scores = [0.0] * n
        for i, s in zip(picked, sub):
            scores[i] = s
        return scores

    docs = [Counter(_tokens(s)) for s in sentences]
    df: Counter = Counter()
    for d in docs:
        df.update(d.keys())
    idf = {w: math.log((n + 1) / (c + 0.5)) for w, c in df.items()}
    vecs = [{w: tf * idf[w] for w, tf in d.items()} for d in docs]
    norms = [math.sqrt(sum(v * v for v in vec.values())) or 1.0 for vec in vecs]

    sim = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            a, b = vecs[i], vecs[j]
            if len(a) > len(b):
                a, b = b, a
            dot = sum(v * b.get(w, 0.0) for w, v in a.items())
            if dot:
                sim[i][j] = sim[j][i] = dot / (norms[i] * norms[j])

    out_sum = [sum(row) or 1.0 for row in sim]
    scores = [1.0 / n] * n
    for _ in range(iterations):
        scores = [(1 - damping) / n + damping * sum(sim[j][i] / out_sum[j] * scores[j] for j in range(n))
                  for i in range(n)]
    return scores


def top_sentences(text: str, max_sentences: int | None = None,
                  max_chars: int | None = None) -> list[str]:
    """Best sentences, returned in their original reading order."""
    sentences = dedupe(split_sentences(text))
    if not sentences:
        return []
    scores = rank_sentences(sentences)
    order = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)
    chosen, used = [], 0
    for i in order:
        if max_sentences is not None and len(chosen) >= max_sentences:
            break
        if max_chars is not None and used + len(sentences[i]) > max_chars and chosen:
            continue
        chosen.append(i)
        used += len(sentences[i]) + 1
    return [sentences[i] for i in sorted(chosen)]


def compress(text: str, max_chars: int) -> str:
    """Shrink long text to roughly `max_chars`, keeping its most central sentences."""
    if len(text) <= max_chars:
        return text
    return " ".join(top_sentences(text, max_chars=max_chars))


def keywords(text: str, n: int = 8) -> list[str]:
    counts = Counter(_tokens(text))
    return [w for w, _ in counts.most_common(n)]
