from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from dataclasses import replace
from enum import Enum
from pathlib import Path
from typing import Annotated

from .glossary import (
    GlossaryEntry,
    build_hotwords,
    collect_glossary,
    describe_glossary,
    parse_glossary,
    parse_renames,
)
from .postprocess import (
    apply_renames,
    build_quality_report,
    dedupe_consecutive,
    drop_hallucinations,
    finalize_subtitles,
    kana_ratio,
    normalize_japanese,
    repair_timestamps,
)
from .subtitles import write_srt, write_vtt
from .transcribe import (
    TranscriptionError,
    TranscriptionResult,
    TranscriptSegment,
    download_model,
    resolve_default_model,
    resolve_device_and_compute_type,
    transcribe_file_with_language,
)
from .translate import (
    DEFAULT_TRANSLATION_MODEL,
    TranslationError,
    translate_segments_to_chinese,
    verify_translation_model,
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

# 中文单行最大字数：再长在播放器里就会溢出。
DEFAULT_WRAP_CHARS = 20


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
    model: str = "auto",
    device: str = "auto",
    compute_type: str = "auto",
    translation_model: str = DEFAULT_TRANSLATION_MODEL,
    overwrite: bool = False,
    progress_callback: ProgressCallback | None = None,
    preset: str | None = None,
    beam_size: int | None = None,
    best_of: int | None = None,
    vad_filter: bool | None = None,
    initial_prompt: str | None = None,
    hotwords: str | None = None,
    glossary: Sequence[GlossaryEntry] | None = None,
    rename: str | None = None,
    wrap_chars: int | None = DEFAULT_WRAP_CHARS,
    debug_originals: bool = False,
    debug_dir: Path | None = None,
    offline: bool = False,
    batch_size: int = 1,
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

    resolved_device, resolved_compute_type = resolve_device_and_compute_type(
        device, compute_type
    )
    resolved_model = model if model and model != "auto" else resolve_default_model(resolved_device)
    entries = list(glossary or ())
    effective_hotwords = build_hotwords(hotwords, entries)
    if entries:
        _emit_progress(
            progress_callback,
            stage="validate",
            message=f"{describe_glossary(entries)}，原词同时作为 hotwords 参与识别",
            percent=0.05,
        )
    _emit_progress(
        progress_callback,
        stage="transcribe",
        message=(
            f"准备开始语音识别（{resolved_model} / {resolved_device} / "
            f"{resolved_compute_type}）"
        ),
        percent=0.08,
    )
    result = transcribe_file_with_language(
        input_path,
        language=language,
        model_name=resolved_model,
        device=resolved_device,
        compute_type=resolved_compute_type,
        progress_callback=progress_callback,
        preset=preset,
        beam_size=beam_size,
        best_of=best_of,
        vad_filter=vad_filter,
        initial_prompt=initial_prompt,
        hotwords=effective_hotwords,
        offline=offline,
    )

    warning = _language_sanity_warning(result, requested_language=language)
    if warning:
        _emit_progress(
            progress_callback,
            stage="transcribe",
            message=warning,
            percent=0.56,
        )

    source_segments = _attach_source_text(result.segments)
    source_segments = repair_timestamps(source_segments)
    raw_count = len(source_segments)
    source_segments = drop_hallucinations(source_segments, language=result.language)
    source_segments = dedupe_consecutive(source_segments)
    cleaned_count = len(source_segments)

    _emit_progress(
        progress_callback,
        stage="translate",
        message="准备翻译为中文",
        percent=0.58,
    )
    translated_segments = translate_segments_to_chinese(
        source_segments,
        source_language=result.language,
        model_path=translation_model,
        device=resolved_device,
        progress_callback=progress_callback,
        glossary=entries,
        batch_size=batch_size,
    )

    renames = parse_renames(rename)
    if renames:
        translated_segments = [
            replace(segment, text=apply_renames(segment.text, renames) or segment.text)
            for segment in translated_segments
        ]
        _emit_progress(
            progress_callback,
            stage="translate",
            message=f"已按译名表统一 {len(renames)} 处中文写法",
            percent=0.9,
        )

    _emit_progress(
        progress_callback,
        stage="write",
        message="正在整理字幕并写入文件",
        percent=0.92,
    )
    final_segments = finalize_subtitles(
        translated_segments,
        language=result.language,
        max_chars=wrap_chars or DEFAULT_WRAP_CHARS,
    )
    for path in output_paths:
        if path.suffix == ".srt":
            write_srt(path, final_segments, wrap_chars=wrap_chars)
        elif path.suffix == ".vtt":
            write_vtt(path, final_segments, wrap_chars=wrap_chars)

    debug_outputs: list[Path] = []
    if debug_originals:
        debug_outputs = _write_debug_artifacts(
            destination=destination,
            debug_dir=debug_dir,
            stem=input_path.stem,
            source_segments=source_segments,
            translated_segments=translated_segments,
            result=result,
            model=resolved_model,
            device=resolved_device,
            compute_type=resolved_compute_type,
            raw_count=raw_count,
            cleaned_count=cleaned_count,
            final_count=len(final_segments),
            warning=warning,
            glossary=entries,
        )

    _emit_progress(
        progress_callback,
        stage="complete",
        message="中文字幕已生成",
        percent=1.0,
        outputs=[str(path) for path in output_paths],
        debug_outputs=[str(path) for path in debug_outputs],
    )
    return output_paths


def _attach_source_text(segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
    """把识别原文保存到 source_text，供翻译/诊断使用。

    翻译阶段会用中文覆盖 ``text``，原文一旦丢失就无法再对照检查。
    """
    return [
        replace(segment, text=normalize_japanese(segment.text), source_text=segment.text)
        for segment in segments
    ]


def _language_sanity_warning(
    result: TranscriptionResult,
    *,
    requested_language: str,
) -> str | None:
    if requested_language != "auto" or not result.segments:
        return None
    joined = "".join(segment.text for segment in result.segments)
    if result.language in {"zh", "yue"} and kana_ratio(joined) > 0.1:
        return (
            "提示：检测语言为中文，但识别结果含大量假名，"
            "源语言可能判断错误；如为日语素材请手动指定 ja。"
        )
    probability = result.language_probability
    if isinstance(probability, int | float) and probability < 0.6:
        return f"提示：语言检测置信度仅 {probability:.0%}，如结果异常请手动指定源语言。"
    return None


def _write_debug_artifacts(
    *,
    destination: Path,
    debug_dir: Path | None,
    stem: str,
    source_segments: list[TranscriptSegment],
    translated_segments: list[TranscriptSegment],
    result: TranscriptionResult,
    model: str,
    device: str,
    compute_type: str,
    raw_count: int,
    cleaned_count: int,
    final_count: int,
    warning: str | None,
    glossary: Sequence[GlossaryEntry] = (),
) -> list[Path]:
    """中间诊断产物：原文 SRT + 双语 SRT + 逐条 QA 报告。

    默认不生成；仅在 ``--debug-originals``（或 UI 勾选诊断输出）时写入，
    用于人工核对"错在识别还是错在翻译"。
    """
    target = (debug_dir or (destination / "debug")).expanduser()
    target.mkdir(parents=True, exist_ok=True)

    ja_path = target / f"{stem}.ja.srt"
    bilingual_path = target / f"{stem}.bilingual.srt"
    qa_path = target / f"{stem}.qa.jsonl"

    write_srt(ja_path, source_segments)
    write_srt(bilingual_path, translated_segments, wrap_chars=DEFAULT_WRAP_CHARS, bilingual=True)

    meta = {
        "input": stem,
        "model": model,
        "device": device,
        "compute_type": compute_type,
        "language": result.language,
        "language_probability": result.language_probability,
        "duration": result.duration,
        "segments_raw": raw_count,
        "segments_after_filter": cleaned_count,
        "segments_final": final_count,
        "warning": warning,
        "glossary": [{"source": entry.source, "target": entry.target} for entry in glossary],
    }
    rows = build_quality_report(translated_segments, language=result.language)
    with qa_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"meta": meta}, ensure_ascii=False) + "\n")
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return [ja_path, bilingual_path, qa_path]


def _collect_glossary(
    inline: str | None,
    path: Path | None,
) -> list[GlossaryEntry]:
    """兼容旧调用点的薄封装。"""
    return collect_glossary(inline, path)


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
            typer.Option(
                "--model",
                "-m",
                help="faster-whisper model name or path; 'auto' picks large-v3 on GPU, small on CPU.",
            ),
        ] = "auto",
        device: Annotated[
            str,
            typer.Option(
                "--device",
                "-d",
                help="Inference device: auto (default), cuda or cpu.",
            ),
        ] = "auto",
        compute_type: Annotated[
            str,
            typer.Option(
                "--compute-type",
                help="Compute type: auto (default), float16, int8_float16 or int8.",
            ),
        ] = "auto",
        translation_model: Annotated[
            str,
            typer.Option(
                "--translation-model",
                help=(
                    "Local GGUF translation model: a .gguf file or a directory containing it "
                    "(default: models/qwen2.5-7b-instruct-q4_k_m)."
                ),
            ),
        ] = DEFAULT_TRANSLATION_MODEL,
        overwrite: Annotated[
            bool,
            typer.Option("--overwrite", help="Replace existing subtitle files."),
        ] = False,
        preset: Annotated[
            str | None,
            typer.Option(
                "--preset",
                help=(
                    "Transcription preset: 'ja' (日语优化), 'noisy' (嘈杂/人声弱), "
                    "'accurate' (高精度), 'default' (平衡) or 'fast' (快速)."
                ),
            ),
        ] = None,
        beam_size: Annotated[
            int | None,
            typer.Option("--beam-size", help="Beam size for decoding (higher = more accurate, slower)."),
        ] = None,
        best_of: Annotated[
            int | None,
            typer.Option("--best-of", help="Candidates sampled when temperature > 0."),
        ] = None,
        no_vad: Annotated[
            bool,
            typer.Option("--no-vad", help="Disable the VAD (Voice Activity Detection) filter."),
        ] = False,
        initial_prompt: Annotated[
            str | None,
            typer.Option(
                "--initial-prompt",
                help="Text hint passed to Whisper. Prefer --hotwords for domain terms.",
            ),
        ] = None,
        hotwords: Annotated[
            str | None,
            typer.Option(
                "--hotwords",
                help="Comma separated names/terms to bias recognition (person names, jargon).",
            ),
        ] = None,
        glossary: Annotated[
            str | None,
            typer.Option(
                "--glossary",
                help=(
                    "Inline glossary, e.g. 'ヒカル=光,みえ=美惠'. Terms are also fed to "
                    "Whisper as hotwords and forced to the given Chinese spelling."
                ),
            ),
        ] = None,
        glossary_file: Annotated[
            Path | None,
            typer.Option("--glossary-file", help="UTF-8 glossary file, one '原词=中文' per line."),
        ] = None,
        rename: Annotated[
            str | None,
            typer.Option(
                "--rename",
                help=(
                    "Chinese-only name fixups, e.g. '希卡尔=光'. Applied to the translated text, "
                    "so no Japanese knowledge is required."
                ),
            ),
        ] = None,
        debug_originals: Annotated[
            bool,
            typer.Option(
                "--debug-originals",
                help="Also write source-language SRT, bilingual SRT and a QA report (diagnostics only).",
            ),
        ] = False,
        offline: Annotated[
            bool,
            typer.Option("--offline", help="Use only locally cached models, never touch the network."),
        ] = False,
        batch_size: Annotated[
            int,
            typer.Option(
                "--batch-size",
                help="Translate N lines per request (1 = one request per line); falls back to per-line when a batch misaligns.",
            ),
        ] = 1,
    ) -> None:
        try:
            entries = _collect_glossary(glossary, glossary_file)
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
                hotwords=hotwords,
                glossary=entries,
                rename=rename,
                debug_originals=debug_originals,
                offline=offline,
                batch_size=batch_size,
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
    parser.add_argument("--model", "-m", default="auto")
    parser.add_argument("--device", "-d", default="auto")
    parser.add_argument("--compute-type", default="auto")
    parser.add_argument("--translation-model", default=DEFAULT_TRANSLATION_MODEL)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--preset", choices=["ja", "noisy", "accurate", "default", "fast"], default=None)
    parser.add_argument("--beam-size", type=int, default=None)
    parser.add_argument("--best-of", type=int, default=None)
    parser.add_argument("--no-vad", action="store_true")
    parser.add_argument("--initial-prompt", default=None)
    parser.add_argument("--hotwords", default=None)
    parser.add_argument("--glossary", default=None, help="inline 原词=中文, e.g. 'ヒカル=光,みえ=美惠'")
    parser.add_argument("--glossary-file", type=Path, default=None)
    parser.add_argument("--rename", default=None, help="中文译名替换, e.g. '希卡尔=光'")
    parser.add_argument("--debug-originals", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Translate N lines per request (1 = one request per line).",
    )
    args = parser.parse_args()

    try:
        entries = _collect_glossary(args.glossary, args.glossary_file)
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
            hotwords=args.hotwords,
            glossary=entries,
            rename=args.rename,
            debug_originals=args.debug_originals,
            offline=args.offline,
            batch_size=args.batch_size,
        )
    except (ValueError, FileExistsError, TranscriptionError, TranslationError) as exc:
        parser.exit(1, f"Error: {exc}\n")

    for output in outputs:
        print(f"Wrote {output}")


def download_model_main() -> None:
    parser = argparse.ArgumentParser(
        prog="subgen-download-model",
        description="Prepare the faster-whisper model and verify the local GGUF translation model.",
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
        translation_path = (
            None
            if args.skip_translation_model
            else verify_translation_model(args.translation_model)
        )
    except (TranscriptionError, TranslationError) as exc:
        parser.exit(1, f"Error: {exc}\n")

    print(f"Whisper model is ready: {args.model} ({args.device})")
    if translation_path is not None:
        print(f"Translation model is ready: {translation_path}")


if __name__ == "__main__":
    main()
