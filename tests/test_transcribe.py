import sys
import types
from pathlib import Path

from subgen.transcribe import TranscriptSegment, download_model, transcribe_file_with_language


def test_download_model_loads_whisper_model(monkeypatch):
    calls = []

    class FakeWhisperModel:
        def __init__(self, model_name, *, device, compute_type):
            calls.append((model_name, device, compute_type))

    fake_module = types.SimpleNamespace(WhisperModel=FakeWhisperModel)
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)

    download_model(model_name="small", device="cpu")

    assert calls == [("small", "cpu", "default")]


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
            return [FakeSegment()], types.SimpleNamespace(language="en")

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
    assert result.segments == [TranscriptSegment(0.5, 1.5, "Hello.")]
