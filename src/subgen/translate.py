from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from pathlib import Path
from time import monotonic

from .transcribe import TranscriptSegment

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_TRANSLATION_MODEL_DIR = PROJECT_ROOT / "models" / "nllb-200-distilled-600M"
HF_TRANSLATION_MODEL_ID = "facebook/nllb-200-distilled-600M"


def resolve_default_translation_model() -> str:
    env_path = os.environ.get("SUBGEN_TRANSLATION_MODEL")
    if env_path:
        return env_path
    if LOCAL_TRANSLATION_MODEL_DIR.is_dir():
        return str(LOCAL_TRANSLATION_MODEL_DIR)
    return HF_TRANSLATION_MODEL_ID


DEFAULT_TRANSLATION_MODEL = resolve_default_translation_model()
TARGET_LANGUAGE = "zho_Hans"

NLLB_LANGUAGE_CODES = {
    "ar": "arb_Arab",
    "de": "deu_Latn",
    "en": "eng_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "hi": "hin_Deva",
    "it": "ita_Latn",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "pt": "por_Latn",
    "ru": "rus_Cyrl",
    "th": "tha_Thai",
    "vi": "vie_Latn",
    "yue": "yue_Hant",
    "zh": "zho_Hans",
}

CHINESE_LANGUAGE_CODES = {
    "yue",
    "zh",
    "zh-cn",
    "zh-hans",
    "zh-hant",
    "zh-tw",
    "zho_hans",
    "zho_hant",
}


class TranslationError(RuntimeError):
    """Raised when local Chinese translation cannot be completed."""


ProgressCallback = Callable[[dict], None]


def download_translation_model(
    *,
    model_name: str = DEFAULT_TRANSLATION_MODEL,
    device: str = "cpu",
    progress_callback: ProgressCallback | None = None,
) -> None:
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ModuleNotFoundError as exc:
        raise TranslationError(
            "Missing translation dependencies. Install them with "
            '`python -m pip install -e ".[translate]"` and try again.'
        ) from exc

    if device == "cuda" and not torch.cuda.is_available():
        raise TranslationError(
            "CUDA was selected, but PyTorch cannot access a CUDA GPU."
        )

    try:
        _emit_progress(
            progress_callback,
            stage="load_translation_model",
            message="正在加载中文翻译模型",
            percent=0.55,
        )
        AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        if device == "cuda":
            model.to("cuda")
        _emit_progress(
            progress_callback,
            stage="translation_model_ready",
            message="中文翻译模型已准备好",
            percent=0.9,
        )
    except Exception as exc:  # pragma: no cover - exercised by real model/runtime failures.
        raise TranslationError(
            "Translation model download or load failed. Check your network, "
            f"model name, and local cache. Cause: {exc}"
        ) from exc


def translate_segments_to_chinese(
    segments: Iterable[TranscriptSegment],
    *,
    source_language: str | None,
    model_name: str = DEFAULT_TRANSLATION_MODEL,
    device: str = "cpu",
    progress_callback: ProgressCallback | None = None,
) -> list[TranscriptSegment]:
    prepared_segments = list(segments)
    if not prepared_segments:
        _emit_progress(
            progress_callback,
            stage="translate",
            message="没有可翻译的字幕片段",
            percent=0.8,
            current=0,
            total=0,
        )
        return []

    if _is_chinese_language(source_language):
        _emit_progress(
            progress_callback,
            stage="translate",
            message="源语言已是中文，跳过翻译",
            percent=0.85,
            current=len(prepared_segments),
            total=len(prepared_segments),
        )
        return prepared_segments

    source_code = _to_nllb_language_code(source_language)
    texts = [segment.text for segment in prepared_segments]
    translated_texts = _translate_texts(
        texts,
        source_code=source_code,
        model_name=model_name,
        device=device,
        progress_callback=progress_callback,
    )

    return [
        TranscriptSegment(start=segment.start, end=segment.end, text=text)
        for segment, text in zip(prepared_segments, translated_texts, strict=True)
    ]


def _translate_texts(
    texts: list[str],
    *,
    source_code: str,
    model_name: str,
    device: str,
    progress_callback: ProgressCallback | None = None,
) -> list[str]:
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ModuleNotFoundError as exc:
        raise TranslationError(
            "Missing translation dependencies. Install them with "
            '`python -m pip install -e ".[translate]"` and try again.'
        ) from exc

    if device == "cuda" and not torch.cuda.is_available():
        raise TranslationError(
            "CUDA was selected, but PyTorch cannot access a CUDA GPU."
        )

    try:
        _emit_progress(
            progress_callback,
            stage="load_translation_model",
            message="正在加载中文翻译模型",
            percent=0.6,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=source_code)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        use_cuda = device == "cuda"
        if use_cuda:
            model = model.to("cuda")

        translated: list[str] = []
        total = len(texts)
        translation_started_at = monotonic()
        _emit_progress(
            progress_callback,
            stage="translate",
            message=f"正在翻译字幕 0 / {total}",
            percent=0.65,
            current=0,
            total=total,
            eta_seconds=None,
        )
        for batch in _chunks(texts, size=8):
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            if use_cuda:
                inputs = {key: value.to("cuda") for key, value in inputs.items()}
            generated = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids(TARGET_LANGUAGE),
                max_new_tokens=256,
            )
            translated.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
            current = len(translated)
            elapsed_seconds = monotonic() - translation_started_at
            eta_seconds = _estimate_remaining_seconds(
                elapsed_seconds=elapsed_seconds,
                current=current,
                total=total,
            )
            _emit_progress(
                progress_callback,
                stage="translate",
                message=f"正在翻译字幕 {current} / {total}",
                percent=0.65 + (0.25 * current / max(total, 1)),
                current=current,
                total=total,
                elapsed_seconds=elapsed_seconds,
                eta_seconds=eta_seconds,
            )
        return translated
    except Exception as exc:  # pragma: no cover - exercised by real model/runtime failures.
        raise TranslationError(
            "Translation failed. Check the source language, translation model, "
            f"local cache, and selected device. Cause: {exc}"
        ) from exc


def _to_nllb_language_code(language: str | None) -> str:
    if language is None or language == "auto":
        raise TranslationError(
            "Source language was not detected, so it cannot be translated to Chinese."
        )
    normalized = language.strip()
    if normalized in NLLB_LANGUAGE_CODES.values():
        return normalized
    short_code = normalized.lower().split("-")[0]
    try:
        return NLLB_LANGUAGE_CODES[short_code]
    except KeyError as exc:
        raise TranslationError(
            f"Unsupported source language for Chinese translation: {language}"
        ) from exc


def _is_chinese_language(language: str | None) -> bool:
    if language is None:
        return False
    normalized = language.strip().replace("_", "-").lower()
    return normalized in CHINESE_LANGUAGE_CODES or normalized.replace("-", "_") in CHINESE_LANGUAGE_CODES


def _chunks(items: list[str], *, size: int) -> Iterable[list[str]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]


def _estimate_remaining_seconds(
    *,
    elapsed_seconds: float,
    current: int,
    total: int,
) -> float | None:
    if current <= 0 or total <= 0 or current >= total:
        return None
    seconds_per_item = elapsed_seconds / current
    return max(0.0, seconds_per_item * (total - current))


def _emit_progress(
    progress_callback: ProgressCallback | None,
    **payload,
) -> None:
    if progress_callback is not None:
        progress_callback(payload)
