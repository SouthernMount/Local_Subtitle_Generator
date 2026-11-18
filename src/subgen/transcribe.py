"""语音识别层：设备解析、语言感知预设、真实进度上报。"""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TranscriptSegment:
    start: float
    end: float
    text: str
    source_text: str | None = None


@dataclass(frozen=True)
class TranscriptionResult:
    segments: list[TranscriptSegment]
    language: str | None
    language_probability: float | None = None
    duration: float | None = None


class TranscriptionError(RuntimeError):
    """Raised when local transcription cannot be completed."""


ProgressCallback = Callable[[dict], None]


# ---------------------------------------------------------------------------
# 运行设备与模型解析
# ---------------------------------------------------------------------------
DEFAULT_CUDA_MODEL = "large-v3"
DEFAULT_CPU_MODEL = "small"
CUDA_COMPUTE_TYPE = "float16"
CPU_COMPUTE_TYPE = "int8"


@functools.lru_cache(maxsize=1)
def cuda_device_count() -> int:
    """检测 CTranslate2 可见的 CUDA 设备数量；缺少依赖时视为 0。"""
    try:
        import ctranslate2
    except ModuleNotFoundError:
        return 0
    try:
        return int(ctranslate2.get_cuda_device_count())
    except Exception:  # pragma: no cover - 驱动异常时保守回退到 CPU
        return 0


def resolve_device_and_compute_type(
    device: str | None,
    compute_type: str | None,
) -> tuple[str, str]:
    """把 ``auto``/``default`` 解析成具体的 device 与 compute_type。

    - ``device='auto'``：有可用的 CTranslate2 CUDA 设备就用 cuda，否则 cpu。
    - ``compute_type='auto'`` 或 ``'default'``：CUDA 用 float16，CPU 用 int8。

    显式传入的值原样保留，便于用户强制指定。
    """
    resolved_device = (device or "auto").strip().lower()
    if resolved_device in {"auto", ""}:
        resolved_device = "cuda" if cuda_device_count() > 0 else "cpu"

    resolved_compute = (compute_type or "auto").strip().lower()
    if resolved_compute in {"auto", "default", ""}:
        resolved_compute = CUDA_COMPUTE_TYPE if resolved_device == "cuda" else CPU_COMPUTE_TYPE
    return resolved_device, resolved_compute


def resolve_default_model(device: str = "auto") -> str:
    """按可用设备选择默认识别模型：GPU 用 large-v3，CPU 用 small。"""
    resolved_device, _ = resolve_device_and_compute_type(device, "auto")
    return DEFAULT_CUDA_MODEL if resolved_device == "cuda" else DEFAULT_CPU_MODEL


# ---------------------------------------------------------------------------
# 预设配置 — 针对不同场景优化的参数组合
# ---------------------------------------------------------------------------
# 日语对话停顿短、句尾助词多，默认 2 秒的静音门槛会把相邻句子切碎。
# max_speech_duration_s 给连续"疑似语音"（音乐、环境音）设上限：否则 VAD 会把
# 20 分钟的 BGM 当成一整段语音，导致 Whisper 输出的时间戳被拉长到分钟级。
JAPANESE_VAD_PARAMETERS: dict[str, Any] = {
    "threshold": 0.5,
    "min_silence_duration_ms": 500,
    "speech_pad_ms": 200,
    "max_speech_duration_s": 30.0,
}

# 嘈杂场景（背景音乐/环境音大、人声弱）：实测把 VAD 门槛从 0.5 降到 0.3，
# 在 0 dB 信噪比素材上能多识别出约 50% 的内容；再配合更长的 speech_pad，
# 避免弱人声的起音/尾音被切掉。门槛不能再降：0.2 以下会把音乐气息当人声，
# 幻觉段反而变多。
NOISY_VAD_PARAMETERS: dict[str, Any] = {
    "threshold": 0.30,
    "min_silence_duration_ms": 400,
    "speech_pad_ms": 300,
    "max_speech_duration_s": 30.0,
}


@dataclass(frozen=True)
class TranscriptionPreset:
    """Parameter preset for different quality/speed trade-offs."""

    beam_size: int = 5
    best_of: int = 5
    temperature: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    vad_filter: bool = True
    vad_parameters: dict[str, Any] | None = None
    word_timestamps: bool = True
    condition_on_previous_text: bool = False
    compression_ratio_threshold: float | None = 2.2
    log_prob_threshold: float | None = -1.0
    no_speech_threshold: float | None = 0.6
    repetition_penalty: float = 1.0
    hallucination_silence_threshold: float | None = 2.0


# 日语专用：更细的 VAD、开启词级时间戳以便抑制幻觉并智能断句。
PRESET_JAPANESE = TranscriptionPreset(
    beam_size=5,
    best_of=5,
    vad_filter=True,
    vad_parameters=JAPANESE_VAD_PARAMETERS,
    word_timestamps=True,
    condition_on_previous_text=False,
    compression_ratio_threshold=2.2,
    log_prob_threshold=-1.0,
    no_speech_threshold=0.65,
    repetition_penalty=1.0,
    hallucination_silence_threshold=2.0,
)

# 嘈杂场景：背景音大、人声偏弱的素材（酒吧、街头、车内、环境音盖住对白）。
# 相比日语预设主要做三件事：
#   1. VAD 门槛降到 0.30，让弱人声也能触发；speech_pad 加长，保住起音/尾音；
#   2. no_speech/log_prob 阈值放宽，避免把听不清但确实有内容的片段直接丢弃；
#   3. compression_ratio 放宽到 2.4，减少"噪声导致重试、重试导致空段"的连锁。
PRESET_NOISY = TranscriptionPreset(
    beam_size=5,
    best_of=5,
    vad_filter=True,
    vad_parameters=NOISY_VAD_PARAMETERS,
    word_timestamps=True,
    condition_on_previous_text=False,
    compression_ratio_threshold=2.4,
    log_prob_threshold=-1.2,
    no_speech_threshold=0.55,
    repetition_penalty=1.0,
    hallucination_silence_threshold=2.0,
)

# 高精度：更宽的 beam，代价约 2 倍耗时。
PRESET_ACCURATE = TranscriptionPreset(
    beam_size=10,
    best_of=5,
    vad_filter=True,
    vad_parameters=JAPANESE_VAD_PARAMETERS,
    word_timestamps=True,
    condition_on_previous_text=False,
    compression_ratio_threshold=2.2,
    log_prob_threshold=-1.0,
    no_speech_threshold=0.6,
    repetition_penalty=1.0,
    hallucination_silence_threshold=2.0,
)

# 平衡：默认质量档，适合英语/中文等主流语言。
# VAD 门槛同样采用细粒度设置：2 秒静音门槛会把正常的句子间隔当成一段，
# 导致字幕条过长、断句不合理，这一点对中文/英语同样成立。
PRESET_DEFAULT = TranscriptionPreset(vad_parameters=JAPANESE_VAD_PARAMETERS)

# 快速：牺牲精度换速度。
PRESET_FAST = TranscriptionPreset(
    beam_size=1,
    best_of=1,
    temperature=(0.0,),
    vad_filter=False,
    vad_parameters=None,
    word_timestamps=False,
    condition_on_previous_text=False,
    compression_ratio_threshold=2.4,
    log_prob_threshold=-1.0,
    no_speech_threshold=0.6,
    repetition_penalty=1.0,
    hallucination_silence_threshold=None,
)

_PRESETS: dict[str, TranscriptionPreset] = {
    "accurate": PRESET_ACCURATE,
    "default": PRESET_DEFAULT,
    "fast": PRESET_FAST,
    "ja": PRESET_JAPANESE,
    "noisy": PRESET_NOISY,
}

# 这些语言的默认语音节奏更适合日语预设（短停顿、防止复读）。
_JAPANESE_LANGUAGES = {"ja", "jp", "jpn"}


def _resolve_preset(
    preset: str | None,
    language: str | None,
    beam_size: int | None,
    best_of: int | None,
    vad_filter: bool | None,
) -> TranscriptionPreset:
    """Merge a named preset with any explicit overrides.

    未指定预设时按源语言选择：日语走日语预设，其他语言走默认预设。
    """
    if preset is not None:
        base = _PRESETS.get(preset)
        if base is None:
            raise ValueError(
                f"Unknown transcription preset: {preset!r}. "
                f"Valid presets: {', '.join(sorted(_PRESETS))}"
            )
    elif (language or "").lower() in _JAPANESE_LANGUAGES:
        base = PRESET_JAPANESE
    else:
        base = PRESET_DEFAULT

    return TranscriptionPreset(
        beam_size=beam_size if beam_size is not None else base.beam_size,
        best_of=best_of if best_of is not None else base.best_of,
        temperature=base.temperature,
        vad_filter=vad_filter if vad_filter is not None else base.vad_filter,
        vad_parameters=base.vad_parameters,
        word_timestamps=base.word_timestamps,
        condition_on_previous_text=base.condition_on_previous_text,
        compression_ratio_threshold=base.compression_ratio_threshold,
        log_prob_threshold=base.log_prob_threshold,
        no_speech_threshold=base.no_speech_threshold,
        repetition_penalty=base.repetition_penalty,
        hallucination_silence_threshold=base.hallucination_silence_threshold,
    )


def _build_initial_prompt(initial_prompt: str | None) -> str | None:
    """Only an explicit prompt is forwarded to the model.

    历史上这里会自动注入一段日语寒暄词表，实测会诱发复读和幻觉，
    因此改为默认不注入；需要领域词汇时请使用 ``hotwords``。
    """
    if initial_prompt is None:
        return None
    prompt = initial_prompt.strip()
    return prompt or None


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

    device, compute_type = resolve_device_and_compute_type(device, compute_type)
    if not model_name or model_name == "auto":
        model_name = resolve_default_model(device)
    try:
        _emit_progress(
            progress_callback,
            stage="load_whisper_model",
            message=f"正在加载识别模型 {model_name}（{device} / {compute_type}）",
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
            f"local cache, and selected device. Cause: {exc}"
        ) from exc


def transcribe_file(
    input_path: Path,
    *,
    language: str = "auto",
    model_name: str = "small",
    device: str = "auto",
    compute_type: str = "auto",
    progress_callback: ProgressCallback | None = None,
    preset: str | None = None,
    beam_size: int | None = None,
    best_of: int | None = None,
    vad_filter: bool | None = None,
    initial_prompt: str | None = None,
    hotwords: str | None = None,
    offline: bool = False,
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
        hotwords=hotwords,
        offline=offline,
    ).segments


def transcribe_file_with_language(
    input_path: Path,
    *,
    language: str = "auto",
    model_name: str = "auto",
    device: str = "auto",
    compute_type: str = "auto",
    progress_callback: ProgressCallback | None = None,
    preset: str | None = None,
    beam_size: int | None = None,
    best_of: int | None = None,
    vad_filter: bool | None = None,
    initial_prompt: str | None = None,
    hotwords: str | None = None,
    offline: bool = False,
) -> TranscriptionResult:
    """Transcribe one local media file and include the detected source language.

    Parameters
    ----------
    model_name:
        Model name or path. ``\"auto\"``/``None`` 时按设备自动选择：
        CUDA 用 ``large-v3``，CPU 用 ``small``。
    device / compute_type:
        ``\"auto\"`` 时自动检测 CUDA 并选择 float16 / int8。
    preset:
        ``\"ja\"``（日语优化）、``\"noisy\"``（嘈杂/人声弱）、``\"accurate\"``（高精度）、
        ``\"default\"``、``\"fast\"``。不指定时按 ``language`` 推断：日语用日语预设。
    hotwords:
        领域词汇/人名提示（faster-whisper 的 hotwords），比 initial_prompt 更可控。
    offline:
        仅使用本地缓存，不访问网络。
    """
    try:
        from faster_whisper import WhisperModel
    except ModuleNotFoundError as exc:
        raise TranscriptionError(
            "Missing dependency 'faster-whisper'. Install the project with "
            "`python -m pip install -e .` and try again."
        ) from exc

    device, compute_type = resolve_device_and_compute_type(device, compute_type)
    if not model_name or model_name == "auto":
        model_name = resolve_default_model(device)

    try:
        _emit_progress(
            progress_callback,
            stage="load_whisper_model",
            message=f"正在加载识别模型 {model_name}（{device} / {compute_type}）",
            percent=0.1,
        )
        model_kwargs: dict[str, Any] = {"device": device, "compute_type": compute_type}
        if offline:
            model_kwargs["local_files_only"] = True
        model = WhisperModel(model_name, **model_kwargs)
        whisper_language = None if language == "auto" else language

        params = _resolve_preset(preset, language, beam_size, best_of, vad_filter)
        effective_prompt = _build_initial_prompt(initial_prompt)

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
            temperature=list(params.temperature),
            vad_filter=params.vad_filter,
            vad_parameters=params.vad_parameters,
            word_timestamps=params.word_timestamps,
            condition_on_previous_text=params.condition_on_previous_text,
            compression_ratio_threshold=params.compression_ratio_threshold,
            log_prob_threshold=params.log_prob_threshold,
            no_speech_threshold=params.no_speech_threshold,
            repetition_penalty=params.repetition_penalty,
            hallucination_silence_threshold=params.hallucination_silence_threshold,
            initial_prompt=effective_prompt,
            hotwords=(hotwords.strip() or None) if hotwords else None,
            language_detection_threshold=0.6,
            language_detection_segments=3,
        )
        detected_language = getattr(info, "language", None) or whisper_language
        detected_probability = getattr(info, "language_probability", None)
        duration = float(getattr(info, "duration", 0.0) or 0.0)
        prepared_segments: list[TranscriptSegment] = []
        for index, segment in enumerate(segments, start=1):
            prepared_segments.append(
                TranscriptSegment(
                    start=float(segment.start),
                    end=float(segment.end),
                    text=str(segment.text),
                )
            )
            if index == 1 or index % 5 == 0:
                _emit_progress(
                    progress_callback,
                    stage="transcribe",
                    message=f"正在识别音频，已获得 {index} 个字幕片段",
                    percent=_transcription_percent(segment.end, duration),
                    current=index,
                )
        _emit_progress(
            progress_callback,
            stage="transcribed",
            message=(
                f"识别完成，检测到源语言：{detected_language or '未知'}"
                + (
                    f"（置信度 {detected_probability:.0%}）"
                    if isinstance(detected_probability, int | float)
                    else ""
                )
            ),
            percent=0.55,
            total=len(prepared_segments),
        )
        return TranscriptionResult(
            segments=prepared_segments,
            language=detected_language,
            language_probability=(
                float(detected_probability)
                if isinstance(detected_probability, int | float)
                else None
            ),
            duration=duration or None,
        )
    except Exception as exc:  # pragma: no cover - exercised by real model/runtime failures.
        raise TranscriptionError(
            "Transcription failed. Check that the media file is readable, the model "
            "can be downloaded or loaded locally, and the selected device is available. "
            f"Cause: {exc}"
        ) from exc


def _transcription_percent(position: float, duration: float) -> float:
    if duration <= 0:
        return 0.35
    ratio = max(0.0, min(1.0, position / duration))
    return 0.25 + 0.3 * ratio


def _emit_progress(
    progress_callback: ProgressCallback | None,
    **payload,
) -> None:
    if progress_callback is not None:
        progress_callback(payload)
