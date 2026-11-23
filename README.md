# 📖 Vidya Offline

An AI tutor that runs **100% on the student's own computer** - no internet, no cloud API, no data leaving the device.
Upload a textbook (PDF / Word / PowerPoint / text / photo of a page) and then:

* **Ask questions** about it - answers come *only* from the book, with page-number citations
* get a **chapter summary**, **flashcards**, a **mock quiz** and a **mind map**, written at the student's class level
* use it in **Hindi, Kannada, Tamil, Telugu, Malayalam, Marathi, Bengali, Gujarati, Punjabi, Odia, Assamese, Urdu**

Built for the Societal Project (Review I deck) - same 5-stage architecture, with DeepSeek-R1 replaced by lighter Ollama models.

---------------------------------------------------------------------------------------------------

## 1. What you need

| | Minimum (works, slower) | Comfortable |
|---|---|---|
| OS | Windows 10/11 64-bit | Windows 10/11 |
| RAM | 4 GB | 8 GB |
| Free disk | 6 GB | 10 GB |
| CPU | any 64-bit, 2+ cores (no GPU needed) | 4+ cores |

Software to install (all free):

1. **Python 3.11 or 3.12** - https://www.python.org/downloads/  → in the installer tick **"Add python.exe to PATH"**.
   (3.12 is what this project was tested on. Avoid 3.13+ for now - some packages lag behind.)
2. **Ollama** - https://ollama.com/download → install and leave it running (tray icon).
3. *(Optional)* Git. Not required - you can just unzip this folder.

You do **not** need: Visual Studio, CUDA, a GPU, Docker, Jupyter, or Graphviz.

## 2. Install - the easy way (Windows)

1. Unzip the folder, e.g. to `C:\vidya-offline`.
2. Double-click **`setup_windows.bat`**. It will:
   * create a private Python environment (`.venv`)
   * install the packages in `requirements.txt` (~500 MB)
   * ask whether to install **offline translation** (Section 5) - say **Y** if you can (~3 GB download, once)
   * run `ollama pull nomic-embed-text` and `ollama pull llama3.2`
   * run a self-check and tell you exactly what is missing
3. Double-click **`run.bat`** → the app opens at http://localhost:8501.

## 3. Install - manual (if you prefer typing)

```bat
cd C:\vidya-offline
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt

ollama pull nomic-embed-text      :: 274 MB  - turns text into searchable vectors
ollama pull llama3.2              :: 2.0 GB  - the tutor brain (3B).   For 4 GB PCs use:  ollama pull llama3.2:1b

python scripts\check_setup.py     :: everything should say [ OK ]
streamlit run app.py
```

Every later session is just: make sure Ollama is running → `run.bat`.

## 4. Pick the right model for the PC

| PC | Chat model | Notes |
|---|---|---|
| 4 GB RAM | `llama3.2:1b` (~1.3 GB) | Fast enough; quiz/flashcard JSON is a bit less reliable - use "Generate" again if it errors |
| 6-8 GB RAM | `llama3.2` (3B, ~2 GB) | Best balance. This is what you already tested in Jupyter |
| 8 GB+, want better Indic/JSON | `qwen2.5:3b` or `qwen2.5:1.5b` | Strong multilingual + structured output |
| Very weak PC | `smollm2:360m` | The architecture diagram's tiny model; low quality, but runs anywhere. Use the *Instant key points* button (no model) as well |
| Also works | `deepseek-r1:1.5b` | `<think>` text is stripped automatically. Slower because it "reasons" first |

Model tag names change over time - browse https://ollama.com/library. Switch models any time in the sidebar (no restart).
Smaller **Context window** (sidebar → Advanced) = less RAM.

## 5. Offline translation (Hindi, Kannada, ...)

The app translates **your question → English** (so retrieval works) and **the answer → your language**, using Meta's
**NLLB-200** model, run locally through CTranslate2 (int8, ~0.7 GB RAM).

```bat
.venv\Scripts\activate
pip install -r requirements-translation.txt
python scripts\download_translation_model.py --slim
```

* Needs internet **once** (≈3 GB download) and ~3 GB free RAM while converting.
* On a 4 GB PC: run those two commands on another computer and **copy the `models` folder** across. After that you may
  `pip uninstall torch` on the school PC.
* Without it the app falls back to asking the chat model to translate - fine for Hindi, **poor for Kannada and other
  languages**. The sidebar shows which engine is active.
* Licence: NLLB-200 weights are **CC-BY-NC 4.0** (non-commercial). Fine for a college project / school use; check
  before any commercial release.

## 6. Using the app

1. **Sidebar → Add textbooks → ⚡ Process & index.** A 300-page PDF takes a few minutes on a weak PC (progress bar shown).
   Scanned pages are read with OCR automatically. Re-uploading the same file is detected - nothing is duplicated.
2. Choose **Student level** (Class 1-3 … College) and **Language**.
3. **💬 Ask my books** - ask anything. Under each answer: 📚 *Sources* with file, page and match distance.
4. **📋 Summary / 🗂️ Flashcards / 📝 Mock quiz / 🧠 Mind map** - pick a book, then *whole book*, a *page range* or a *topic*.
5. Flashcards download as CSV (Anki-ready); summaries as Markdown.

**How the tutor decides where an answer comes from (3-layer routing):**

1. Greeting / small talk → answered directly.
2. Question matches the book → answered **only** from the retrieved passages, with citations.
3. No match → answered from the model's general knowledge **with a visible warning** ("check with your teacher").

### Tuning the "is it in the book?" limit
Embedding models differ, so the default limit (0.65) may need adjusting for your books:

```bat
python scripts\calibrate.py "What is photosynthesis?" "Who won the 2011 cricket world cup?"
```
Pick a limit above the distances of questions that *should* match and below those that *shouldn't*, then set the
sidebar slider **Book-match strictness** (or `VIDYA_MAX_DISTANCE`).

## 7. How it maps to the architecture slide

| Stage | In the slide | In this project | File |
|---|---|---|---|
| 1 Local inputs | PDF, DOCX, images, camera photos | pypdf, python-docx, python-pptx, Pillow | `vidya/ingest.py` |
| 2 Lightweight processing | MobileNet PP-OCR (~20 MB), LanceDB | **RapidOCR (PP-OCR via ONNX)**, sentence-aware chunking, **LanceDB** | `ingest.py`, `store.py`, `library.py`, `pipeline.py` |
| 3 Reasoning & NLP | SmolLM2 / Llama 3.2 on CPU + TextRank | **Ollama** (any model), 3-layer router, **TextRank** | `llm.py`, `rag.py`, `textrank.py`, `embeddings.py` |
| 4 Translation | INMT-Lite / MarianMT | **NLLB-200 600M** (CTranslate2 int8), LLM fallback | `translate.py` |
| 5 Interactive outputs | Dashboard: summaries, mind maps, flashcards, quizzes | Streamlit dashboard | `app.py`, `study.py` |

```
vidya-offline/
├─ app.py                      Streamlit dashboard (Stage 5)
├─ vidya/                      the engine (no UI code - reusable from FastAPI / the MasterG app later)
│  ├─ config.py  llm.py  embeddings.py  store.py  library.py  pipeline.py
│  ├─ ingest.py  textrank.py  rag.py  study.py  translate.py
├─ scripts/                    check_setup.py · calibrate.py · download_translation_model.py
├─ tests/                      39 automated tests (pytest)
├─ data/                       created at run time: your books + vector index (delete to reset)
├─ models/                     optional: NLLB translation model
├─ setup_windows.bat  run.bat
└─ requirements*.txt
```

## 8. Settings (environment variables, all optional)

`VIDYA_CHAT_MODEL` (default `llama3.2`) · `VIDYA_EMBED_MODEL` (`nomic-embed-text`) · `VIDYA_NUM_CTX` (4096) ·
`VIDYA_TOP_K` (4) · `VIDYA_MAX_DISTANCE` (0.65) · `VIDYA_CHUNK_SIZE` (900) · `VIDYA_OCR` (`0` disables OCR) ·
`VIDYA_DATA_DIR` · `VIDYA_MODELS_DIR` · `VIDYA_OLLAMA_HOST`. Example: `set VIDYA_CHAT_MODEL=llama3.2:1b` before `run.bat`.

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| "Ollama is not running" | Start the Ollama app (or `ollama serve`); reload the page |
| "Required models are missing" | Run the `ollama pull ...` lines the page shows |
| `pip install` fails on a package | Check `python --version` is 3.10-3.12; `python -m pip install --upgrade pip`; retry |
| Every question says "No matching passage" | Run `scripts\calibrate.py`, raise **Book-match strictness**; check the book really contains text (scans need OCR on) |
| Unrelated questions get book answers | Lower **Book-match strictness** |
| Very slow / PC freezes | Use `llama3.2:1b`, lower Context window to 2048, close the browser tabs and other programs |
| Quiz / flashcards "did not return valid JSON" | Click again (it retries); use a bigger model or a shorter page range |
| Hindi/Kannada looks poor | Install NLLB (Section 5) - sidebar must say "NLLB-200 …" |
| Changed embedding model → empty answers | Sidebar shows "not indexed" → press **Re-index** (no re-upload needed) |
| Start over | Close the app, delete the `data` folder |

Run `python scripts\check_setup.py` at any time for a full diagnosis.

## 10. Honest limitations

* Small models make mistakes. Answers are grounded and cited, but a 1-3 B model can still misread a passage -
  teachers should treat it as a study helper, not an authority. Check the 📚 Sources.
* The interface text is English; only *content* (questions, answers, summaries, quizzes, mind maps) is translated.
* Machine translation of technical terms can be awkward; Kannada and other low-resource languages are weaker than Hindi.
* No drawing whiteboard and no mobile app in this version - `vidya/` is UI-free so it can be wrapped in FastAPI and
  reused by a React-Native/Expo front end.
* "Offline" means *after setup*. Installing Python packages, Ollama and the models needs internet once; the finished
  folder (`.venv`, `models`, plus `%USERPROFILE%\.ollama`) can be copied to other PCs.

## 11. Tests

```bat
pip install -r requirements-dev.txt
python -m pytest tests -q
```
Covers ingestion (incl. scanned-PDF OCR), chunking, LanceDB search/filter/delete, routing, quiz validation, the real
`ollama` client against a fake server, and headless UI runs of every tab.
