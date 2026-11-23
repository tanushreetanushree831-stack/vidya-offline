"""Stage 4: offline translation into Indian languages.

Backends, best first:
  ct2  NLLB-200-distilled-600M converted to CTranslate2 int8  (~0.7 GB RAM, fast on CPU)  <- recommended
  hf   NLLB-200-distilled-600M with PyTorch                    (~2.5 GB RAM)
  llm  the local Ollama chat model                              (no extra download, weak for Indic languages)
"""
from __future__ import annotations

import importlib.util
import os
import re
from pathlib import Path

LANGUAGES: dict[str, str] = {
    "English": "eng_Latn",
    "Hindi (हिन्दी)": "hin_Deva",
    "Kannada (ಕನ್ನಡ)": "kan_Knda",
    "Tamil (தமிழ்)": "tam_Taml",
    "Telugu (తెలుగు)": "tel_Telu",
    "Malayalam (മലയാളം)": "mal_Mlym",
    "Marathi (मराठी)": "mar_Deva",
    "Bengali (বাংলা)": "ben_Beng",
    "Gujarati (ગુજરાતી)": "guj_Gujr",
    "Punjabi (ਪੰਜਾਬੀ)": "pan_Guru",
    "Odia (ଓଡ଼ିଆ)": "ory_Orya",
    "Assamese (অসমীয়া)": "asm_Beng",
    "Urdu (اردو)": "urd_Arab",
}
ENGLISH = "eng_Latn"
_CODE_TO_NAME = {v: k.split(" (")[0] for k, v in LANGUAGES.items()}

_SENT = re.compile(r"(?<=[.!?\u0964\u0965])\s+")
_MD_PREFIX = re.compile(r"^(\s*(?:#{1,6}\s+|[-*\u2022]\s+|\d+[.)]\s+))")


def looks_non_english(text: str) -> bool:
    return any(c.isalpha() and ord(c) > 0x024F for c in text)


def _has(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


class Translator:
    def __init__(self, ct2_dir: Path, hf_dir: Path, llm=None) -> None:
        self.ct2_dir, self.hf_dir, self.llm = Path(ct2_dir), Path(hf_dir), llm
        self._cache: dict[tuple, str] = {}
        self._loaded: str | None = None
        self._tok = None
        self._model = None

    # ---- backend discovery ----
    def _has_tokenizer(self) -> bool:
        return (self.hf_dir / "tokenizer_config.json").exists() or (self.hf_dir / "sentencepiece.bpe.model").exists()

    def _has_hf_weights(self) -> bool:
        return any((self.hf_dir / f).exists() for f in ("model.safetensors", "pytorch_model.bin"))

    def backend_name(self) -> str:
        if self.ct2_dir.exists() and self._has_tokenizer() and _has("ctranslate2") and _has("transformers"):
            return "ct2"
        if self._has_hf_weights() and _has("transformers") and _has("torch"):
            return "hf"
        return "llm" if self.llm is not None else "none"

    def describe(self) -> str:
        return {"ct2": "NLLB-200 (CTranslate2 int8, offline)",
                "hf": "NLLB-200 (PyTorch, offline)",
                "llm": "Local chat model only - install NLLB for much better Indian-language quality",
                "none": "No translation engine available"}[self.backend_name()]

    # ---- loading ----
    def _load(self) -> str:
        if self._loaded:
            return self._loaded
        backend = self.backend_name()
        if backend == "ct2":
            import ctranslate2
            import transformers
            self._tok = transformers.AutoTokenizer.from_pretrained(str(self.hf_dir), local_files_only=True)
            self._model = ctranslate2.Translator(str(self.ct2_dir), device="cpu", compute_type="int8",
                                                 inter_threads=1, intra_threads=max(1, (os.cpu_count() or 4) - 1))
        elif backend == "hf":
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(str(self.hf_dir), local_files_only=True)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(str(self.hf_dir), local_files_only=True)
        self._loaded = backend
        return backend

    # ---- core ----
    def _nllb_batch(self, batch: list[str], src: str, tgt: str, backend: str) -> list[str]:
        tok = self._tok
        tok.src_lang = src
        if backend == "ct2":
            toks = [tok.convert_ids_to_tokens(tok.encode(s)) for s in batch]
            res = self._model.translate_batch(toks, target_prefix=[[tgt]] * len(batch),
                                              beam_size=2, max_decoding_length=256)
            return [tok.decode(tok.convert_tokens_to_ids(r.hypotheses[0][1:]), skip_special_tokens=True)
                    for r in res]
        inputs = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=256)
        out = self._model.generate(**inputs, forced_bos_token_id=tok.convert_tokens_to_ids(tgt),
                                   max_new_tokens=256, num_beams=2)
        return tok.batch_decode(out, skip_special_tokens=True)

    def _llm_translate(self, text: str, tgt: str) -> str:
        name = _CODE_TO_NAME.get(tgt, tgt)
        msg = [{"role": "system", "content": f"You are a professional translator. Translate the user's text into {name}. "
                                             "Output ONLY the translation, nothing else."},
               {"role": "user", "content": text}]
        return self.llm.chat(msg, temperature=0.1).strip()

    def translate(self, text: str, src: str, tgt: str) -> str:
        if not text.strip() or src == tgt:
            return text
        key = (src, tgt, text)
        if key in self._cache:
            return self._cache[key]
        backend = self._load()
        if backend == "none":
            raise RuntimeError("No translation engine is available.")
        if backend == "llm":
            result = self._llm_translate(text, tgt) if src == ENGLISH else self._llm_translate(text, ENGLISH)
        else:
            sentences = [s for para in text.split("\n") for s in _SENT.split(para) if s.strip()]
            parts: list[str] = []
            for i in range(0, len(sentences), 8):
                parts.extend(self._nllb_batch(sentences[i:i + 8], src, tgt, backend))
            result = " ".join(parts)
        if len(self._cache) > 2000:
            self._cache.clear()
        self._cache[key] = result
        return result

    def translate_many(self, texts: list[str], src: str, tgt: str) -> list[str]:
        return [self.translate(t, src, tgt) for t in texts]

    def translate_markdown(self, md: str, tgt: str, src: str = ENGLISH) -> str:
        """Translate line by line, keeping headings and bullet markers intact."""
        out = []
        for line in md.splitlines():
            if not line.strip():
                out.append(line)
                continue
            m = _MD_PREFIX.match(line)
            prefix = m.group(1) if m else ""
            body = line[len(prefix):].replace("**", "").replace("__", "")
            out.append(prefix + self.translate(body, src, tgt))
        return "\n".join(out)
