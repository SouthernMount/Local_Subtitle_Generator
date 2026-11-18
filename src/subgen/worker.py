from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import time

from .cli import SubtitleFormat, generate_subtitles
from .glossary import collect_glossary
from .transcribe import TranscriptionError, download_model
from .translate import (
    DEFAULT_TRANSLATION_MODEL,
    TranslationError,
    verify_translation_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m subgen.worker")
    parser.add_argument("--result-file", required=True, type=Path)
    parser.add_argument("--progress-file", type=Path)

    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--input-path", required=True, type=Path)
    generate_parser.add_argument("--output-dir", type=Path)
    generate_parser.add_argument("--format", choices=[item.value for item in SubtitleFormat], default="both")
    generate_parser.add_argument("--language", default="auto")
    generate_parser.add_argument("--model", default="auto")
    generate_parser.add_argument("--device", default="auto")
    generate_parser.add_argument("--compute-type", default="auto")
    generate_parser.add_argument("--translation-model", default=DEFAULT_TRANSLATION_MODEL)
    generate_parser.add_argument("--overwrite", action="store_true")
    # --- 高级参数 ---
    generate_parser.add_argument("--preset", choices=["ja", "noisy", "accurate", "default", "fast"], default=None)
    generate_parser.add_argument("--beam-size", type=int, default=None)
    generate_parser.add_argument("--best-of", type=int, default=None)
    generate_parser.add_argument("--no-vad", action="store_true")
    generate_parser.add_argument("--initial-prompt", default=None)
    generate_parser.add_argument("--hotwords", default=None)
    generate_parser.add_argument("--glossary", default=None)
    generate_parser.add_argument("--glossary-file", type=Path, default=None)
    generate_parser.add_argument("--rename", default=None)
    generate_parser.add_argument("--debug-originals", action="store_true")
    generate_parser.add_argument("--debug-dir", type=Path, default=None)
    generate_parser.add_argument("--offline", action="store_true")
    generate_parser.add_argument("--batch-size", type=int, default=1)

    download_parser = subparsers.add_parser("download")
    download_parser.add_argument("--model", default="small")
    download_parser.add_argument("--device", default="cpu")
    download_parser.add_argument("--compute-type", default="default")
    download_parser.add_argument("--translation-model", default=DEFAULT_TRANSLATION_MODEL)

    args = parser.parse_args()
    progress_callback = _make_progress_callback(args.progress_file, started_at=time())

    try:
        if args.command == "generate":
            progress_callback(
                {
                    "stage": "start",
                    "message": "开始生成中文字幕",
                    "percent": 0.0,
                }
            )
            outputs = generate_subtitles(
                args.input_path,
                output_dir=args.output_dir,
                subtitle_format=SubtitleFormat(args.format),
                language=args.language,
                model=args.model,
                device=args.device,
                compute_type=args.compute_type,
                translation_model=args.translation_model,
                overwrite=args.overwrite,
                progress_callback=progress_callback,
                preset=args.preset,
                beam_size=args.beam_size,
                best_of=args.best_of,
                vad_filter=False if args.no_vad else None,
                initial_prompt=args.initial_prompt,
                hotwords=args.hotwords,
                glossary=collect_glossary(args.glossary, args.glossary_file),
                rename=args.rename,
                debug_originals=args.debug_originals,
                debug_dir=args.debug_dir,
                offline=args.offline,
                batch_size=args.batch_size,
            )
            _write_result(args.result_file, ok=True, outputs=[str(path) for path in outputs])
            return

        if args.command == "download":
            progress_callback(
                {
                    "stage": "start",
                    "message": "开始准备模型",
                    "percent": 0.0,
                }
            )
            download_model(
                model_name=args.model,
                device=args.device,
                compute_type=args.compute_type,
                progress_callback=progress_callback,
            )
            verify_translation_model(args.translation_model)
            progress_callback(
                {
                    "stage": "complete",
                    "message": "模型已准备好",
                    "percent": 1.0,
                }
            )
            _write_result(args.result_file, ok=True, outputs=[])
            return

    except (ValueError, FileExistsError, TranscriptionError, TranslationError) as exc:
        progress_callback(
            {
                "stage": "error",
                "message": str(exc),
                "percent": None,
            }
        )
        _write_result(args.result_file, ok=False, error=str(exc))
        raise SystemExit(1) from exc
    except Exception as exc:  # pragma: no cover - protects the UI from unexpected worker failures.
        progress_callback(
            {
                "stage": "error",
                "message": f"Unexpected worker error: {exc}",
                "percent": None,
            }
        )
        _write_result(args.result_file, ok=False, error=f"Unexpected worker error: {exc}")
        raise SystemExit(1) from exc


def _write_result(
    result_file: Path,
    *,
    ok: bool,
    outputs: list[str] | None = None,
    error: str | None = None,
) -> None:
    result_file.parent.mkdir(parents=True, exist_ok=True)
    result_file.write_text(
        json.dumps(
            {
                "ok": ok,
                "outputs": outputs or [],
                "error": error,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _make_progress_callback(progress_file: Path | None, *, started_at: float):
    def callback(payload: dict) -> None:
        if progress_file is not None:
            _write_progress(progress_file, payload, started_at=started_at)

    return callback


def _write_progress(progress_file: Path, payload: dict, *, started_at: float) -> None:
    now = time()
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    normalized = {
        "stage": payload.get("stage", "running"),
        "message": payload.get("message", ""),
        "percent": payload.get("percent"),
        "current": payload.get("current"),
        "total": payload.get("total"),
        "elapsed_seconds": payload.get("elapsed_seconds", now - started_at),
        "eta_seconds": payload.get("eta_seconds"),
        "outputs": payload.get("outputs", []),
        "debug_outputs": payload.get("debug_outputs", []),
        "started_at": started_at,
        "updated_at": now,
    }
    progress_file.write_text(
        json.dumps(normalized, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
