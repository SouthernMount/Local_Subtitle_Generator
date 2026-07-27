from subgen.ui import (
    _apply_selected_transcription_preset,
    _ensure_transcription_param_state,
    _format_duration,
    _mark_transcription_preset_custom,
)


def test_format_duration() -> None:
    assert _format_duration(4.4) == "4 秒"
    assert _format_duration(65) == "1 分 5 秒"
    assert _format_duration(3661) == "1 小时 1 分 1 秒"


def test_apply_selected_transcription_preset_updates_visible_params() -> None:
    session_state = {"preset_select": "accurate"}

    _apply_selected_transcription_preset(session_state)

    assert session_state["beam_size_input"] == 10
    assert session_state["best_of_input"] == 10
    assert session_state["vad_filter_checkbox"] is True


def test_fast_transcription_preset_updates_visible_params() -> None:
    session_state = {
        "preset_select": "fast",
        "beam_size_input": 10,
        "best_of_input": 10,
        "vad_filter_checkbox": True,
    }

    _apply_selected_transcription_preset(session_state)

    assert session_state["beam_size_input"] == 1
    assert session_state["best_of_input"] == 1
    assert session_state["vad_filter_checkbox"] is False


def test_manual_transcription_param_change_marks_preset_custom() -> None:
    session_state = {"preset_select": "accurate"}

    _mark_transcription_preset_custom(session_state)

    assert session_state["preset_select"] is None


def test_ensure_transcription_param_state_uses_selected_preset_defaults() -> None:
    session_state = {"preset_select": "fast"}

    _ensure_transcription_param_state(session_state)

    assert session_state["beam_size_input"] == 1
    assert session_state["best_of_input"] == 1
    assert session_state["vad_filter_checkbox"] is False
