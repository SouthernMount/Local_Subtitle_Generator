import sys
import types

import pytest

from subgen.transcribe import TranscriptSegment
from subgen.translate import (
    TranslationError,
    _estimate_remaining_seconds,
    download_translation_model,
    translate_segments_to_chinese,
)


def test_translate_skips_chinese_source_language() -> None:
    segments = [TranscriptSegment(0, 1, "已经是中文。")]

    assert translate_segments_to_chinese(segments, source_language="zh") == segments


def test_translate_english_segments_to_chinese(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeTensor:
        def to(self, _device):
            return self

    class FakeInputs(dict):
        def __init__(self):
            super().__init__({"input_ids": FakeTensor()})

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, model_name, *, src_lang=None):
            assert model_name == "fake-model"
            assert src_lang == "eng_Latn"
            return cls()

        def __call__(self, batch, **_kwargs):
            assert batch == ["Hello."]
            return FakeInputs()

        def convert_tokens_to_ids(self, token):
            assert token == "zho_Hans"
            return 1

        def batch_decode(self, _generated, *, skip_special_tokens):
            assert skip_special_tokens is True
            return ["你好。"]

    class FakeModel:
        @classmethod
        def from_pretrained(cls, model_name):
            assert model_name == "fake-model"
            return cls()

        def generate(self, **_kwargs):
            return [[1, 2, 3]]

    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False)
    )
    fake_transformers = types.SimpleNamespace(
        AutoModelForSeq2SeqLM=FakeModel,
        AutoTokenizer=FakeTokenizer,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    translated = translate_segments_to_chinese(
        [TranscriptSegment(0, 1, "Hello.")],
        source_language="en",
        model_name="fake-model",
        device="cpu",
    )

    assert translated == [TranscriptSegment(0, 1, "你好。")]


def test_translate_unsupported_language_has_clear_error() -> None:
    with pytest.raises(TranslationError, match="Unsupported source language"):
        translate_segments_to_chinese(
            [TranscriptSegment(0, 1, "Text.")],
            source_language="xx",
        )


def test_translate_cuda_unavailable_has_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False)
    )
    fake_transformers = types.SimpleNamespace(
        AutoModelForSeq2SeqLM=object,
        AutoTokenizer=object,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    with pytest.raises(TranslationError, match="CUDA was selected"):
        translate_segments_to_chinese(
            [TranscriptSegment(0, 1, "Hello.")],
            source_language="en",
            device="cuda",
        )


def test_estimate_remaining_seconds() -> None:
    assert _estimate_remaining_seconds(elapsed_seconds=20, current=5, total=10) == 20
    assert _estimate_remaining_seconds(elapsed_seconds=20, current=10, total=10) is None
    assert _estimate_remaining_seconds(elapsed_seconds=20, current=0, total=10) is None
