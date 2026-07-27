from __future__ import annotations

import argparse
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import Annotated

from .subtitles import write_srt, write_vtt
from .transcribe import TranscriptionError, download_model, transcribe_file_with_language
from .translate import (
    DEFAULT_TRANSLATION_MODEL,
    TranslationError,
    download_translation_model,
    translate_segments_to_chinese,
)

try:  # pragma: no cover - import branch depends on optional runtime dependency.
    import typer
    from rich.console import Console
except ModuleNotFoundError:  # pragma: no cover - covered through argparse fallback behavior.
    typer = None
    Console = None


class SubtitleFormat(str, Enum):
    srt = "srt"
    vtt = "vtt"
    both = "both"


ProgressCallback = Callable[[dict], None]


if typer is not None:
    app = typer.Typer(
        add_completion=False,
        help="Generate Chinese SRT/VTT subtitles from one video or audio file.",
    )
    console = Console()
else:
    app = None
    console = None


def generate_subtitles(
    input_path: Path,
    *,
    output_dir: Path | None = None,
    subtitle_format: SubtitleFormat = SubtitleFormat.both,
    language: str = "auto",
    model: str = "small",
    device: str = "cpu",
    compute_type: str = "default",
    translation_model: str = DEFAULT_TRANSLATION_MODEL,
    overwrite: bool = False,
    progress_callback: ProgressCallback | None = None,
    # --- 新增高级参数 ---
    preset: str | None = None,
    beam_size: int | None = None,
    best_of: int | None = None,
    vad_filter: bool | None = None,
    initial_prompt: str | None = None,
) -> list[Path]:
    _emit_progress(
        progress_callback,
        stage="validate",
        message="正在检查输入和输出路径",
        percent=0.02,
    )
    input_path = input_path.expanduser().resolve()
    if not input_path.exists():
        raise ValueError(f"Input file does not exist: {input_path}")
    if not input_path.is_file():
        raise ValueError(f"Input path is not a file: {input_path}")

    destination = (
        output_dir.expanduser().resolve() if output_dir is not None else input_path.parent
    )
    destination.mkdir(parents=True, exist_ok=True)

    output_paths = _planned_outputs(input_path, destination, subtitle_format)
    existing_paths = [path for path in output_paths if path.exists()]
    if existing_paths and not overwrite:
        existing = ", ".join(str(path) for path in existing_paths)
        raise FileExistsError(
            f"Output file already exists: {existing}. Pass --overwrite to replace it."
        )

    _emit_progress(
        progress_callback,
        stage="transcribe",
        message="准备开始语音识别",
        percent=0.08,
    )
    result = transcribe_file_with_language(
        input_path,
        language=language,
        model_name=model,
        device=device,
        compute_type=compute_type,
        progress_callback=progress_callback,
        preset=preset,
        beam_size=beam_size,
        best_of=best_of,
        vad_filter=vad_filter,
        initial_prompt=initial_prompt,
    )
    _emit_progress(
        progress_callback,
        stage="translate",
        message="准备翻译为中文",
        percent=0.58,
    )
    segments = translate_segments_to_chinese(
        result.segments,
        source_language=result.language,
        model_name=translation_model,
        device=device,
        progress_callback=progress_callback,
    )

    _emit_progress(
        progress_callback,
        stage="write",
        message="正在写入字幕文件",
        percent=0.92,
    )
    for path in output_paths:
        if path.suffix == ".srt":
            write_srt(path, segments)
        elif path.suffix == ".vtt":
            write_vtt(path, segments)

    _emit_progress(
        progress_callback,
        stage="complete",
        message="中文字幕已生成",
        percent=1.0,
        outputs=[str(path) for path in output_paths],
    )
    return output_paths


def _planned_outputs(
    input_path: Path,
    output_dir: Path,
    subtitle_format: SubtitleFormat,
) -> list[Path]:
    stem = input_path.stem
    if subtitle_format == SubtitleFormat.srt:
        return [output_dir / f"{stem}.srt"]
    if subtitle_format == SubtitleFormat.vtt:
        return [output_dir / f"{stem}.vtt"]
    return [output_dir / f"{stem}.srt", output_dir / f"{stem}.vtt"]


def _emit_progress(
    progress_callback: ProgressCallback | None,
    **payload,
) -> None:
    if progress_callback is not None:
        progress_callback(payload)


if typer is not None:

    @app.command()
    def run(
        input_path: Annotated[Path, typer.Argument(help="Input video or audio file.")],
        output_dir: Annotated[
            Path | None,
            typer.Option("--output-dir", "-o", help="Directory for generated subtitles."),
        ] = None,
        subtitle_format: Annotated[
            SubtitleFormat,
            typer.Option("--format", "-f", help="Subtitle output format."),
        ] = SubtitleFormat.both,
        language: Annotated[
            str,
            typer.Option(
                "--language",
                "-l",
                help="Source language code, or 'auto'. Output is always Chinese.",
            ),
        ] = "auto",
        model: Annotated[
            str,
            typer.Option("--model", "-m", help="faster-whisper model name or path."),
        ] = "small",
        device: Annotated[
            str,
            typer.Option("--device", "-d", help="Device for inference, such as cpu or cuda."),
        ] = "cpu",
        compute_type: Annotated[
            str,
            typer.Option(
                "--compute-type",
                help="faster-whisper compute type, such as default, float16, int8_float16, or int8.",
            ),
        ] = "default",
        translation_model: Annotated[
            str,
            typer.Option(
                "--translation-model",
                help="Local translation model used to output Chinese subtitles.",
            ),
        ] = DEFAULT_TRANSLATION_MODEL,
        overwrite: Annotated[
            bool,
            typer.Option("--overwrite", help="Replace existing subtitle files."),
        ] = False,
        # --- 新增高级参数 ---
        preset: Annotated[
            str | None,
            typer.Option(
                "--preset",
                help=(
                    "Transcription quality preset: 'accurate' (高精度，适合日语等), "
                    "'default' (平衡), or 'fast' (快速)."
                ),
            ),
        ] = None,
        beam_size: Annotated[
            int | None,
            typer.Option(
                "--beam-size",
                help="Beam size for decoding (higher = more accurate, slower). Default: 5 (10 with --preset accurate).",
            ),
        ] = None,
        best_of: Annotated[
            int | None,
            typer.Option(
                "--best-of",
                help="Number of candidates sampled when temperature > 0. Default: 5 (10 with --preset accurate).",
            ),
        ] = None,
        no_vad: Annotated[
            bool,
            typer.Option(
                "--no-vad",
                help="Disable the VAD (Voice Activity Detection) filter.",
            ),
        ] = False,
        initial_prompt: Annotated[
            str | None,
            typer.Option(
                "--initial-prompt",
                help="Text hint to steer Whisper recognition. Has built-in defaults for Japanese.",
            ),
        ] = None,
    ) -> None:
        try:
            outputs = generate_subtitles(
                input_path,
                output_dir=output_dir,
                subtitle_format=subtitle_format,
                language=language,
                model=model,
                device=device,
                compute_type=compute_type,
                translation_model=translation_model,
                overwrite=overwrite,
                preset=preset,
                beam_size=beam_size,
                best_of=best_of,
                vad_filter=False if no_vad else None,
                initial_prompt=initial_prompt,
            )
        except (ValueError, FileExistsError, TranscriptionError, TranslationError) as exc:
            console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(code=1) from exc

        for output in outputs:
            console.print(f"[green]Wrote[/green] {output}")


def main() -> None:
    if typer is not None:
        app()
        return

    parser = argparse.ArgumentParser(
        prog="subgen",
        description="Generate Chinese SRT/VTT subtitles from one video or audio file.",
    )
    parser.add_argument("input_path", type=Path)
    parser.add_argument("--output-dir", "-o", type=Path)
    parser.add_argument("--format", "-f", choices=[item.value for item in SubtitleFormat], default="both")
    parser.add_argument("--language", "-l", default="auto")
    parser.add_argument("--model", "-m", default="small")
    parser.add_argument("--device", "-d", default="cpu")
    parser.add_argument("--compute-type", default="default")
    parser.add_argument("--translation-model", default=DEFAULT_TRANSLATION_MODEL)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--preset", choices=["accurate", "default", "fast"], default=None)
    parser.add_argument("--beam-size", type=int, default=None)
    parser.add_argument("--best-of", type=int, default=None)
    parser.add_argument("--no-vad", action="store_true")
    parser.add_argument("--initial-prompt", default=None)
    args = parser.parse_args()

    try:
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
            preset=args.preset,
            beam_size=args.beam_size,
            best_of=args.best_of,
            vad_filter=False if args.no_vad else None,
            initial_prompt=args.initial_prompt,
        )
    except (ValueError, FileExistsError, TranscriptionError, TranslationError) as exc:
        parser.exit(1, f"Error: {exc}\n")

    for output in outputs:
        print(f"Wrote {output}")


def download_model_main() -> None:
    parser = argparse.ArgumentParser(
        prog="subgen-download-model",
        description="Download or load a faster-whisper model without transcribing media.",
    )
    parser.add_argument("--model", "-m", default="small")
    parser.add_argument("--device", "-d", default="cpu")
    parser.add_argument("--compute-type", default="default")
    parser.add_argument("--translation-model", default=DEFAULT_TRANSLATION_MODEL)
    parser.add_argument(
        "--skip-translation-model",
        action="store_true",
        help="Only download or load the faster-whisper model.",
    )
    args = parser.parse_args()

    try:
        download_model(
            model_name=args.model,
            device=args.device,
            compute_type=args.compute_type,
        )
        if not args.skip_translation_model:
            download_translation_model(
                model_name=args.translation_model,
                device=args.device,
            )
    except (TranscriptionError, TranslationError) as exc:
        parser.exit(1, f"Error: {exc}\n")

    print(f"Whisper model is ready: {args.model} ({args.device})")
    if not args.skip_translation_model:
        print(f"Translation model is ready: {args.translation_model}")


if __name__ == "__main__":
    main()
