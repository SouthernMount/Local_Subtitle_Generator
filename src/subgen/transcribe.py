from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class TranscriptSegment:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class TranscriptionResult:
    segments: list[TranscriptSegment]
    language: str | None


class TranscriptionError(RuntimeError):
    """Raised when local transcription cannot be completed."""


ProgressCallback = Callable[[dict], None]


# ---------------------------------------------------------------------------
# 预设配置 — 针对不同场景优化的参数组合
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TranscriptionPreset:
    """Parameter preset for different quality/speed trade-offs."""

    beam_size: int = 5
    best_of: int = 5
    temperature: list[float] = field(default_factory=lambda: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    vad_filter: bool = True
    word_timestamps: bool = False
    condition_on_previous_text: bool = True
    compression_ratio_threshold: float = 2.4
    log_prob_threshold: float = -1.0
    no_speech_threshold: float = 0.6
    repetition_penalty: float = 1.0


# 预设：高精度（适合日语等 Whisper 原生识别率偏低的语言）
PRESET_ACCURATE = TranscriptionPreset(
    beam_size=10,
    best_of=10,
    temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    vad_filter=True,
    word_timestamps=False,
    condition_on_previous_text=True,
    compression_ratio_threshold=2.4,
    log_prob_threshold=-1.0,
    no_speech_threshold=0.6,
    repetition_penalty=1.0,
)

# 预设：平衡（默认，适合英语/中文等主流语言）
PRESET_DEFAULT = TranscriptionPreset(
    beam_size=5,
    best_of=5,
    temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    vad_filter=True,
    word_timestamps=False,
    condition_on_previous_text=True,
    compression_ratio_threshold=2.4,
    log_prob_threshold=-1.0,
    no_speech_threshold=0.6,
    repetition_penalty=1.0,
)

# 预设：快速（牺牲精度换速度）
PRESET_FAST = TranscriptionPreset(
    beam_size=1,
    best_of=1,
    temperature=[0.0],
    vad_filter=False,
    word_timestamps=False,
    condition_on_previous_text=False,
    compression_ratio_threshold=2.4,
    log_prob_threshold=-1.0,
    no_speech_threshold=0.6,
    repetition_penalty=1.0,
)

_PRESETS: dict[str, TranscriptionPreset] = {
    "accurate": PRESET_ACCURATE,
    "default": PRESET_DEFAULT,
    "fast": PRESET_FAST,
}

# ---------------------------------------------------------------------------
# 日语 initial_prompt — 引导模型更好地识别日语内容
# ---------------------------------------------------------------------------
JAPANESE_INITIAL_PROMPT = (
    "こんにちは、お願いします。ありがとうございます。"
    "私は、あなた、それ、この、その、あの、"
    "日本語、会話、番組、ニュース、"
    "はい、いいえ、わかりました、すみません。"
)

# 各语言默认的 initial_prompt（可被用户显式覆盖）
DEFAULT_INITIAL_PROMPTS: dict[str, str] = {
    "ja": JAPANESE_INITIAL_PROMPT,
}


def download_model(
    *,
    model_name: str = "small",
    device: str = "cpu",
    compute_type: str = "default",
    progress_callback: ProgressCallback | None = None,
) -> None:
    """Download or load a faster-whisper model without transcribing media."""
    try:
        from faster_whisper import WhisperModel
    except ModuleNotFoundError as exc:
        raise TranscriptionError(
            "Missing dependency 'faster-whisper'. Install the project with "
            "`python -m pip install -e .` and try again."
        ) from exc

    try:
        _emit_progress(
            progress_callback,
            stage="load_whisper_model",
            message="正在加载识别模型",
            percent=0.15,
        )
        WhisperModel(model_name, device=device, compute_type=compute_type)
        _emit_progress(
            progress_callback,
            stage="whisper_model_ready",
            message="识别模型已准备好",
            percent=0.5,
        )
    except Exception as exc:  # pragma: no cover - exercised by real model/runtime failures.
        raise TranscriptionError(
            "Model download or load failed. Check your network, model name, "
            "local cache, and selected device."
        ) from exc


def _resolve_preset(
    preset: str | None,
    beam_size: int | None,
    best_of: int | None,
    vad_filter: bool | None,
) -> TranscriptionPreset:
    """Merge a named preset with any explicit overrides."""
    if preset is not None:
        base = _PRESETS.get(preset)
        if base is None:
            raise ValueError(
                f"Unknown transcription preset: {preset!r}. "
                f"Valid presets: {', '.join(sorted(_PRESETS))}"
            )
    else:
        base = PRESET_DEFAULT

    # Allow individual parameters to override the preset
    return TranscriptionPreset(
        beam_size=beam_size if beam_size is not None else base.beam_size,
        best_of=best_of if best_of is not None else base.best_of,
        temperature=base.temperature,
        vad_filter=vad_filter if vad_filter is not None else base.vad_filter,
        word_timestamps=base.word_timestamps,
        condition_on_previous_text=base.condition_on_previous_text,
        compression_ratio_threshold=base.compression_ratio_threshold,
        log_prob_threshold=base.log_prob_threshold,
        no_speech_threshold=base.no_speech_threshold,
        repetition_penalty=base.repetition_penalty,
    )


def _build_initial_prompt(
    language: str | None,
    initial_prompt: str | None,
) -> str | None:
    """Determine the effective initial_prompt."""
    if initial_prompt is not None:
        return initial_prompt
    if language is not None and language in DEFAULT_INITIAL_PROMPTS:
        return DEFAULT_INITIAL_PROMPTS[language]
    return None


def transcribe_file(
    input_path: Path,
    *,
    language: str = "zh",
    model_name: str = "small",
    device: str = "cpu",
    compute_type: str = "default",
    progress_callback: ProgressCallback | None = None,
    # --- 新增高级参数 ---
    preset: str | None = None,
    beam_size: int | None = None,
    best_of: int | None = None,
    vad_filter: bool | None = None,
    initial_prompt: str | None = None,
) -> list[TranscriptSegment]:
    """Transcribe one local media file with faster-whisper."""
    return transcribe_file_with_language(
        input_path,
        language=language,
        model_name=model_name,
        device=device,
        compute_type=compute_type,
        progress_callback=progress_callback,
        preset=preset,
        beam_size=beam_size,
        best_of=best_of,
        vad_filter=vad_filter,
        initial_prompt=initial_prompt,
    ).segments


def transcribe_file_with_language(
    input_path: Path,
    *,
    language: str = "zh",
    model_name: str = "small",
    device: str = "cpu",
    compute_type: str = "default",
    progress_callback: ProgressCallback | None = None,
    # --- 新增高级参数 ---
    preset: str | None = None,
    beam_size: int | None = None,
    best_of: int | None = None,
    vad_filter: bool | None = None,
    initial_prompt: str | None = None,
) -> TranscriptionResult:
    """Transcribe one local media file and include the detected source language.

    Parameters
    ----------
    preset:
        Named parameter preset: ``"accurate"``, ``"default"``, or ``"fast"``.
        Individual parameters (``beam_size`` etc.) override the preset.
    beam_size:
        Beam size for decoding (higher = more accurate but slower).
    best_of:
        Number of candidates when ``temperature > 0`` (higher = more accurate).
    vad_filter:
        Enable the Silero VAD filter to remove silent sections before inference.
    initial_prompt:
        Text hint passed to Whisper to steer recognition toward expected vocabulary.
        When ``None`` and ``language`` is ``"ja"``, a built-in Japanese prompt is used.
    """
    try:
        from faster_whisper import WhisperModel
    except ModuleNotFoundError as exc:
        raise TranscriptionError(
            "Missing dependency 'faster-whisper'. Install the project with "
            "`python -m pip install -e .` and try again."
        ) from exc

    try:
        _emit_progress(
            progress_callback,
            stage="load_whisper_model",
            message="正在加载识别模型",
            percent=0.1,
        )
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
        whisper_language = None if language == "auto" else language

        # 解析参数预设
        params = _resolve_preset(preset, beam_size, best_of, vad_filter)
        effective_prompt = _build_initial_prompt(whisper_language, initial_prompt)

        _emit_progress(
            progress_callback,
            stage="transcribe",
            message="正在识别音频，识别阶段无法提供精确百分比",
            percent=0.25,
        )
        segments, info = model.transcribe(
            str(input_path),
            language=whisper_language,
            beam_size=params.beam_size,
            best_of=params.best_of,
            temperature=params.temperature,
            vad_filter=params.vad_filter,
            word_timestamps=params.word_timestamps,
            condition_on_previous_text=params.condition_on_previous_text,
            compression_ratio_threshold=params.compression_ratio_threshold,
            log_prob_threshold=params.log_prob_threshold,
            no_speech_threshold=params.no_speech_threshold,
            repetition_penalty=params.repetition_penalty,
            initial_prompt=effective_prompt,
        )
        detected_language = getattr(info, "language", None) or whisper_language
        prepared_segments = []
        for index, segment in enumerate(segments, start=1):
            prepared_segments.append(
                TranscriptSegment(
                    start=float(segment.start),
                    end=float(segment.end),
                    text=str(segment.text),
                )
            )
            if index == 1 or index % 10 == 0:
                _emit_progress(
                    progress_callback,
                    stage="transcribe",
                    message=f"正在识别音频，已获得 {index} 个字幕片段",
                    percent=0.35,
                    current=index,
                )
        _emit_progress(
            progress_callback,
            stage="transcribed",
            message=f"识别完成，检测到源语言：{detected_language or '未知'}",
            percent=0.55,
            total=len(prepared_segments),
        )
        return TranscriptionResult(
            segments=prepared_segments,
            language=detected_language,
        )
    except Exception as exc:  # pragma: no cover - exercised by real model/runtime failures.
        raise TranscriptionError(
            "Transcription failed. Check that the media file is readable, the model "
            "can be downloaded or loaded locally, and the selected device is available."
        ) from exc


def _emit_progress(
    progress_callback: ProgressCallback | None,
    **payload,
) -> None:
    if progress_callback is not None:
        progress_callback(payload)
