"""One-time download + conversion of NLLB-200 (distilled 600M) for OFFLINE translation.

Needs internet ONCE and  pip install -r requirements-translation.txt
    python scripts/download_translation_model.py --slim

Result in ./models:
    nllb-200-distilled-600M/        tokenizer (+ weights unless --slim)
    nllb-200-distilled-600M-ct2/    int8 CTranslate2 model (~600 MB) used by the app

Converting needs ~3 GB of free RAM. On a 4 GB PC, run this on another computer and simply copy
the whole 'models' folder across (it is plain files).

Licence note: NLLB-200 weights are CC-BY-NC 4.0 (non-commercial use only).
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vidya.config import load_config  # noqa: E402

MODEL = "facebook/nllb-200-distilled-600M"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slim", action="store_true", help="delete the PyTorch weights after converting (saves ~2.4 GB)")
    ap.add_argument("--no-ct2", action="store_true", help="skip CTranslate2 conversion (use PyTorch at runtime)")
    args = ap.parse_args()
    cfg = load_config()

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    cfg.nllb_hf_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading tokenizer and model {MODEL} ...")
    AutoTokenizer.from_pretrained(MODEL).save_pretrained(cfg.nllb_hf_dir)
    AutoModelForSeq2SeqLM.from_pretrained(MODEL).save_pretrained(cfg.nllb_hf_dir)

    if not args.no_ct2:
        from ctranslate2.converters import TransformersConverter
        print("Converting to CTranslate2 int8 ...")
        TransformersConverter(str(cfg.nllb_hf_dir)).convert(str(cfg.nllb_ct2_dir), quantization="int8", force=True)
        if args.slim:
            for f in ("model.safetensors", "pytorch_model.bin"):
                (cfg.nllb_hf_dir / f).unlink(missing_ok=True)
            print("Removed PyTorch weights (tokenizer kept).")
    print("\nDone. Restart the app and pick a language in the sidebar.")


if __name__ == "__main__":
    main()
