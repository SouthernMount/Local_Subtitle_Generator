import sys
import types
from pathlib import Path

import pytest

from subgen import transcribe
from subgen.transcribe import (
    TranscriptSegment,
    download_model,
    resolve_default_model,
    resolve_device_and_compute_type,
    transcribe_file_with_language,
)


def test_download_model_loads_whisper_model(monkeypatch):
    calls = []

    class FakeWhisperModel:
        def __init__(self, model_name, *, device, compute_type):
            calls.append((model_name, device, compute_type))

    fake_module = types.SimpleNamespace(WhisperModel=FakeWhisperModel)
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)

    download_model(model_name="small", device="cpu")

    # CPU 上 ``default`` 会被解析成 int8，与运行时保持一致。
    assert calls == [("small", "cpu", "int8")]


def test_transcribe_file_with_language_returns_detected_language(monkeypatch, tmp_path):
    media_path = tmp_path / "sample.mp4"
    media_path.write_bytes(b"placeholder")

    class FakeSegment:
        start = 0.5
        end = 1.5
        text = "Hello."

    class FakeWhisperModel:
        def __init__(self, _model_name, *, device, compute_type):
            assert device == "cpu"
            assert compute_type == "float16"

        def transcribe(self, input_path, *, language, **kwargs):
            assert input_path == str(media_path)
            assert language is None
            # Verify new parameters are passed through with defaults
            assert kwargs.get("beam_size") == 5
            assert kwargs.get("best_of") == 5
            assert kwargs.get("vad_filter") is True
            assert kwargs.get("word_timestamps") is True
            return [FakeSegment()], types.SimpleNamespace(
                language="en", language_probability=0.98, duration=2.0
            )

    fake_module = types.SimpleNamespace(WhisperModel=FakeWhisperModel)
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)

    result = transcribe_file_with_language(
        Path(media_path),
        language="auto",
        model_name="small",
        device="cpu",
        compute_type="float16",
    )

    assert result.language == "en"
    assert result.language_probability == pytest.approx(0.98)
    assert result.duration == pytest.approx(2.0)
    assert result.segments == [TranscriptSegment(0.5, 1.5, "Hello.")]


def test_auto_device_prefers_cuda_and_float16(monkeypatch):
    monkeypatch.setattr(transcribe, "cuda_device_count", lambda: 1)

    assert resolve_device_and_compute_type("auto", "auto") == ("cuda", "float16")
    assert resolve_default_model("auto") == "large-v3"


def test_auto_device_falls_back_to_cpu_and_int8(monkeypatch):
    monkeypatch.setattr(transcribe, "cuda_device_count", lambda: 0)

    assert resolve_device_and_compute_type("auto", "auto") == ("cpu", "int8")
    assert resolve_default_model("auto") == "small"


def test_explicit_device_and_compute_type_are_preserved(monkeypatch):
    monkeypatch.setattr(transcribe, "cuda_device_count", lambda: 1)

    assert resolve_device_and_compute_type("cpu", "float16") == ("cpu", "float16")


def test_japanese_language_selects_japanese_preset(monkeypatch, tmp_path):
    captured = {}

    class FakeWhisperModel:
        def __init__(self, _model_name, *, device, compute_type):
            pass

        def transcribe(self, _input_path, **kwargs):
            captured.update(kwargs)
            return [], types.SimpleNamespace(language="ja", duration=10.0)

    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=FakeWhisperModel))

    media_path = tmp_path / "sample.mp4"
    media_path.write_bytes(b"placeholder")
    transcribe_file_with_language(Path(media_path), language="ja", model_name="small", device="cpu")

    assert captured["vad_parameters"]["min_silence_duration_ms"] == 500
    assert captured["vad_parameters"]["max_speech_duration_s"] == pytest.approx(30.0)
    assert captured["word_timestamps"] is True
    assert captured["hallucination_silence_threshold"] == pytest.approx(2.0)
    assert captured["condition_on_previous_text"] is False
    assert captured["language_detection_segments"] == 3


def test_initial_prompt_is_not_injected_automatically(monkeypatch, tmp_path):
    captured = {}

    class FakeWhisperModel:
        def __init__(self, _model_name, *, device, compute_type):
            pass

        def transcribe(self, _input_path, **kwargs):
            captured.update(kwargs)
            return [], types.SimpleNamespace(language="ja", duration=1.0)

    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=FakeWhisperModel))

    media_path = tmp_path / "sample.mp4"
    media_path.write_bytes(b"placeholder")
    transcribe_file_with_language(Path(media_path), language="ja", model_name="small", device="cpu")

    assert captured["initial_prompt"] is None
    assert captured["hotwords"] is None


def test_offline_flag_uses_local_files_only(monkeypatch, tmp_path):
    captured = {}

    class FakeWhisperModel:
        def __init__(self, _model_name, **kwargs):
            captured.update(kwargs)

        def transcribe(self, _input_path, **_kwargs):
            return [], types.SimpleNamespace(language="ja", duration=1.0)

    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=FakeWhisperModel))

    media_path = tmp_path / "sample.mp4"
    media_path.write_bytes(b"placeholder")
    transcribe_file_with_language(
        Path(media_path),
        language="ja",
        model_name="small",
        device="cpu",
        offline=True,
    )

    assert captured["local_files_only"] is True


def test_noisy_preset_lowers_vad_threshold() -> None:
    params = transcribe._resolve_preset("noisy", "ja", None, None, None)

    assert params.vad_parameters is not None
    assert params.vad_parameters["threshold"] == pytest.approx(0.30)
    assert params.vad_parameters["speech_pad_ms"] == 300
    assert params.no_speech_threshold == pytest.approx(0.55)


def test_noisy_preset_is_applied_regardless_of_language() -> None:
    params = transcribe._resolve_preset("noisy", "en", None, None, None)

    assert params.vad_parameters == transcribe.NOISY_VAD_PARAMETERS

