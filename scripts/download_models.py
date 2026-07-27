"""Download project-local translation models used by Subtitle Generator."""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ID = "facebook/nllb-200-distilled-600M"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "nllb-200-distilled-600M"
REQUIRED_FILES = (
    "config.json",
    "generation_config.json",
    "sentencepiece.bpe.model",
)
WEIGHT_FILES = (
    "pytorch_model.bin",
    "model.safetensors",
)
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the NLLB translation model into the project models directory.",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--force", action="store_true", help="Replace an existing local model directory.")
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and _model_dir_is_complete(output_dir) and not args.force:
        print(f"Translation model already exists: {output_dir}")
        return

    if output_dir.exists() and not args.force:
        sys.exit(
            f"Incomplete translation model directory found: {output_dir}\n"
            "Delete it or rerun with --force."
        )

    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ModuleNotFoundError:
        sys.exit(
            "Missing transformers dependencies. Run setup.bat or install with "
            'python -m pip install -e ".[translate]".'
        )

    temp_dir = output_dir.parent / f".{output_dir.name}.tmp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.model_id} to {output_dir} ...")
    print("This is about 2.4 GB and can take a while on slower networks.")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_id)
        tokenizer.save_pretrained(str(temp_dir))
        model.save_pretrained(str(temp_dir))

        if not _model_dir_is_complete(temp_dir):
            raise RuntimeError("download finished, but required model files are missing")

        if output_dir.exists():
            shutil.rmtree(output_dir)
        temp_dir.replace(output_dir)
    except Exception:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise

    print(f"Translation model is ready: {output_dir}")


def _model_dir_is_complete(path: Path) -> bool:
    if not path.is_dir():
        return False
    has_required = all((path / name).is_file() for name in REQUIRED_FILES)
    has_weight = any((path / name).is_file() for name in WEIGHT_FILES)
    has_tokenizer = any((path / name).is_file() for name in TOKENIZER_FILES)
    return has_required and has_weight and has_tokenizer


if __name__ == "__main__":
    main()
