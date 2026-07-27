from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

if __package__:
    from .cli import SubtitleFormat, generate_subtitles
    from .transcribe import TranscriptionError, download_model
    from .translate import DEFAULT_TRANSLATION_MODEL, TranslationError, download_translation_model
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from subgen.cli import SubtitleFormat, generate_subtitles
    from subgen.transcribe import TranscriptionError, download_model
    from subgen.translate import DEFAULT_TRANSLATION_MODEL, TranslationError, download_translation_model


TRANSCRIPTION_PRESET_UI_PARAMS = {
    None: {"beam_size": 5, "best_of": 5, "vad_filter": True},
    "accurate": {"beam_size": 10, "best_of": 10, "vad_filter": True},
    "default": {"beam_size": 5, "best_of": 5, "vad_filter": True},
    "fast": {"beam_size": 1, "best_of": 1, "vad_filter": False},
}

TRANSCRIPTION_PRESET_LABELS = {
    None: "自定义（使用下方参数）",
    "accurate": "高精度 — 适合日语等较难的语言",
    "default": "平衡 — 适合英语/中文等主流语言",
    "fast": "快速 — 牺牲精度换速度",
}


def _ensure_transcription_param_state(session_state) -> None:
    if "preset_select" not in session_state:
        session_state["preset_select"] = None

    selected_preset = session_state["preset_select"]
    params = TRANSCRIPTION_PRESET_UI_PARAMS.get(selected_preset, TRANSCRIPTION_PRESET_UI_PARAMS[None])
    defaults = {
        "beam_size_input": params["beam_size"],
        "best_of_input": params["best_of"],
        "vad_filter_checkbox": params["vad_filter"],
    }
    for key, value in defaults.items():
        if key not in session_state:
            session_state[key] = value


def _apply_selected_transcription_preset(session_state) -> None:
    selected_preset = session_state.get("preset_select")
    if selected_preset is None:
        return

    params = TRANSCRIPTION_PRESET_UI_PARAMS[selected_preset]
    session_state["beam_size_input"] = params["beam_size"]
    session_state["best_of_input"] = params["best_of"]
    session_state["vad_filter_checkbox"] = params["vad_filter"]


def _mark_transcription_preset_custom(session_state) -> None:
    session_state["preset_select"] = None


def render_app() -> None:
    import streamlit as st

    language_labels = {
        "auto": "自动检测",
        "zh": "中文",
        "en": "英语",
        "ja": "日语",
        "ko": "韩语",
    }
    format_labels = {
        SubtitleFormat.both.value: "SRT + VTT",
        SubtitleFormat.srt.value: "仅 SRT",
        SubtitleFormat.vtt.value: "仅 VTT",
    }
    compute_type_labels = {
        "default": "自动",
        "float16": "float16（GPU 推荐）",
        "int8_float16": "int8_float16（GPU 省显存）",
        "int8": "int8（CPU 推荐）",
    }

    st.set_page_config(
        page_title="本地中文字幕生成器",
        page_icon=None,
        layout="wide",
    )

    st.markdown(
        """
        <style>
        footer, #MainMenu, [data-testid="stDecoration"],
        [data-testid="stStatusWidget"], [data-testid="stDeployButton"] {
            visibility: hidden;
            height: 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("本地中文字幕生成器")
    st.caption("识别视频或音频中的语音，并只输出中文字幕文件。")

    with st.sidebar:
        st.subheader("参数设置")
        model = st.selectbox(
            "识别模型",
            ["tiny", "base", "small", "medium", "large-v3"],
            index=2,
            key="model_select",
        )
        device = st.selectbox(
            "运行设备",
            ["cpu", "cuda"],
            index=0,
            key="device_select",
            format_func=lambda value: value.upper(),
        )
        compute_type = st.selectbox(
            "计算精度",
            ["default", "float16", "int8_float16", "int8"],
            index=0,
            key="compute_type_select",
            format_func=lambda value: compute_type_labels[value],
            help="CUDA 通常选择 float16；显存不足可试 int8_float16；CPU 可用 default 或 int8。",
        )
        if device == "cuda":
            st.info("GPU 模式需要 NVIDIA 显卡、可用 CUDA 驱动，以及支持 CUDA 的 PyTorch/ctranslate2。")
        language = st.selectbox(
            "源语言",
            ["auto", "zh", "en", "ja", "ko"],
            index=0,
            key="language_select",
            format_func=lambda value: f"{language_labels[value]} ({value})",
        )
        subtitle_format = st.radio(
            "字幕格式",
            [SubtitleFormat.both.value, SubtitleFormat.srt.value, SubtitleFormat.vtt.value],
            horizontal=True,
            key="subtitle_format_radio",
            format_func=lambda value: format_labels[value],
        )
        translation_model = st.text_input(
            "中文翻译模型",
            value=DEFAULT_TRANSLATION_MODEL,
            key="translation_model_text",
            help="非中文语音会先识别为源语言文本，再用此本地模型翻译为中文。",
        )
        overwrite = st.checkbox("覆盖已有字幕文件", value=False, key="overwrite_checkbox")

        _ensure_transcription_param_state(st.session_state)
        with st.expander("高级识别参数", expanded=False):
            preset = st.selectbox(
                "参数预设",
                list(TRANSCRIPTION_PRESET_UI_PARAMS),
                key="preset_select",
                format_func=lambda v: TRANSCRIPTION_PRESET_LABELS.get(v, str(v)),
                help="选择一个预设会自动调整下方参数。选择「自定义」则完全使用手动参数。",
                on_change=_apply_selected_transcription_preset,
                args=(st.session_state,),
            )
            col1, col2 = st.columns(2)
            with col1:
                beam_size = st.number_input(
                    "beam_size",
                    min_value=1,
                    max_value=20,
                    value=5,
                    step=1,
                    key="beam_size_input",
                    help="Beam search 宽度。越大越准确但越慢。默认 5，高精度模式 10。",
                    on_change=_mark_transcription_preset_custom,
                    args=(st.session_state,),
                )
            with col2:
                best_of = st.number_input(
                    "best_of",
                    min_value=1,
                    max_value=20,
                    value=5,
                    step=1,
                    key="best_of_input",
                    help="temperature > 0 时的候选数。越大越准确但越慢。默认 5，高精度模式 10。",
                    on_change=_mark_transcription_preset_custom,
                    args=(st.session_state,),
                )
            vad_filter = st.checkbox(
                "启用 VAD 语音活动检测",
                value=True,
                key="vad_filter_checkbox",
                help="用 Silero VAD 模型先过滤静音段，大幅减少 Whisper 的幻觉输出。强烈推荐开启。",
                on_change=_mark_transcription_preset_custom,
                args=(st.session_state,),
            )
            initial_prompt = st.text_area(
                "initial_prompt（引导文本）",
                value="",
                key="initial_prompt_text",
                placeholder="留空则对日语自动使用内置引导词。手动输入可覆盖默认值。",
                help="Whisper 会将此文本作为上文来引导识别方向。日语已内置默认引导词。",
                on_change=_mark_transcription_preset_custom,
                args=(st.session_state,),
            )

        st.divider()
        if st.button("停止服务", use_container_width=True, key="stop_service_button"):
            st.warning("正在停止本地服务，页面稍后会断开连接。")
            threading.Timer(0.8, lambda: os._exit(0)).start()
            st.stop()

    if "input_path_text" not in st.session_state:
        st.session_state.input_path_text = ""

    input_col, choose_col = st.columns([4, 1])
    with choose_col:
        st.write("")
        st.write("")
        if st.button("选择媒体文件", use_container_width=True, key="choose_media_button"):
            selected_path = _select_media_file()
            if selected_path:
                st.session_state.input_path_text = selected_path
                st.rerun()
    with input_col:
        input_path_text = st.text_input(
            "媒体文件路径",
            key="input_path_text",
            placeholder=r"D:\Videos\sample.mp4",
        )

    output_dir_text = st.text_input(
        "输出目录",
        key="output_dir_text",
        placeholder="留空则输出到媒体文件所在目录",
    )

    action_col, model_col = st.columns([1, 1])

    with model_col:
        if st.button("下载识别和翻译模型", use_container_width=True, key="download_models_button"):
            with st.status("正在准备模型", expanded=True) as status:
                progress_bar = st.progress(0)
                progress_detail = st.empty()
                worker_result = _run_worker(
                    [
                        "download",
                        "--model",
                        model,
                        "--device",
                        device,
                        "--compute-type",
                        compute_type,
                        "--translation-model",
                        translation_model,
                    ],
                    progress_bar=progress_bar,
                    progress_detail=progress_detail,
                )
                if not worker_result["ok"]:
                    status.update(label="模型准备失败", state="error")
                    st.error(_to_chinese_error(worker_result["error"]))
                else:
                    status.update(label="模型已准备好", state="complete")

    with action_col:
        generate_clicked = st.button(
            "生成中文字幕",
            type="primary",
            use_container_width=True,
            key="generate_subtitles_button",
        )

    if not generate_clicked:
        return

    input_path = Path(input_path_text.strip()) if input_path_text.strip() else None
    output_dir = Path(output_dir_text.strip()) if output_dir_text.strip() else None

    if input_path is None:
        st.error("请填写媒体文件路径。")
        return

    with st.status("正在生成中文字幕", expanded=True) as status:
        progress_bar = st.progress(0)
        progress_detail = st.empty()
        worker_args = [
            "generate",
            "--input-path",
            str(input_path),
            "--format",
            subtitle_format,
            "--language",
            language,
            "--model",
            model,
            "--device",
            device,
            "--compute-type",
            compute_type,
            "--translation-model",
            translation_model,
        ]
        if output_dir is not None:
            worker_args.extend(["--output-dir", str(output_dir)])
        if overwrite:
            worker_args.append("--overwrite")
        # --- 高级参数 ---
        if preset:
            worker_args.extend(["--preset", preset])
        worker_args.extend(["--beam-size", str(beam_size)])
        worker_args.extend(["--best-of", str(best_of)])
        if not vad_filter:
            worker_args.append("--no-vad")
        if initial_prompt.strip():
            worker_args.extend(["--initial-prompt", initial_prompt.strip()])

        worker_result = _run_worker(
            worker_args,
            progress_bar=progress_bar,
            progress_detail=progress_detail,
        )
        if not worker_result["ok"]:
            status.update(label="生成失败", state="error")
            st.error(_to_chinese_error(worker_result["error"]))
            return

        status.update(label="中文字幕已生成", state="complete")

    for output_text in worker_result["outputs"]:
        output = Path(output_text)
        st.success(str(output))
        st.download_button(
            label=f"下载 {output.suffix.upper().lstrip('.')}",
            data=output.read_bytes(),
            file_name=output.name,
            mime="text/plain",
            use_container_width=True,
            key=f"download_{output.suffix.lower().lstrip('.')}_{abs(hash(str(output)))}",
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="subgen-ui",
        description="启动本地中文字幕生成器可视化界面。",
    )
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--host", default="localhost")
    args = parser.parse_args()

    try:
        import streamlit.web.cli as streamlit_cli
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Streamlit is not installed. Install UI dependencies with "
            '`python -m pip install -e ".[ui]"`.'
        ) from exc

    script_path = Path(__file__).resolve()
    sys.argv = [
        "streamlit",
        "run",
        str(script_path),
        "--server.headless",
        "true",
        "--server.address",
        args.host,
        "--server.port",
        str(args.port),
        "--browser.gatherUsageStats",
        "false",
    ]
    raise SystemExit(streamlit_cli.main())


def _to_chinese_error(exc: Exception) -> str:
    message = str(exc)
    replacements = {
        "Input file does not exist:": "输入文件不存在：",
        "Input path is not a file:": "输入路径不是文件：",
        "Output file already exists:": "输出文件已存在：",
        "Pass --overwrite to replace it.": "请勾选“覆盖已有字幕文件”后重试。",
        "Media file path is required.": "请填写媒体文件路径。",
        "Missing dependency 'faster-whisper'.": "缺少依赖 faster-whisper。",
        "Missing translation dependencies.": "缺少本地翻译依赖。",
        "Transcription failed.": "语音识别失败。",
        "Translation failed.": "中文翻译失败。",
        "Model download or load failed.": "识别模型下载或加载失败。",
        "Translation model download or load failed.": "中文翻译模型下载或加载失败。",
        "CUDA was selected, but PyTorch cannot access a CUDA GPU.": (
            "已选择 CUDA，但 PyTorch 无法访问 CUDA GPU。请检查显卡驱动、CUDA 版 PyTorch 和当前虚拟环境。"
        ),
        "Source language was not detected, so it cannot be translated to Chinese.": (
            "未检测到源语言，因此无法翻译为中文。"
        ),
        "Unsupported source language for Chinese translation:": "暂不支持翻译为中文的源语言：",
        "Check your network, model name, local cache, and selected device.": (
            "请检查网络、模型名称、本地缓存和运行设备。"
        ),
        "Check your network, model name, and local cache.": (
            "请检查网络、模型名称和本地缓存。"
        ),
        "Check the source language, translation model, local cache, and selected device.": (
            "请检查源语言、翻译模型、本地缓存和运行设备。"
        ),
        "Check that the media file is readable, the model can be downloaded or loaded locally, and the selected device is available.": (
            "请确认媒体文件可读取，识别模型可下载或已存在于本地缓存，并且所选运行设备可用。"
        ),
    }
    for source, target in replacements.items():
        message = message.replace(source, target)
    return message


def _run_worker(
    args: list[str],
    *,
    progress_bar=None,
    progress_detail=None,
) -> dict:
    with tempfile.TemporaryDirectory(prefix="subgen-") as temp_dir:
        result_file = Path(temp_dir) / "result.json"
        progress_file = Path(temp_dir) / "progress.json"
        command = [
            sys.executable,
            "-m",
            "subgen.worker",
            "--result-file",
            str(result_file),
            "--progress-file",
            str(progress_file),
            *args,
        ]
        env = os.environ.copy()
        src_path = str(Path(__file__).resolve().parents[1])
        env["PYTHONPATH"] = (
            src_path
            if not env.get("PYTHONPATH")
            else f"{src_path}{os.pathsep}{env['PYTHONPATH']}"
        )
        process = subprocess.Popen(
            command,
            cwd=str(Path(__file__).resolve().parents[2]),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        last_progress = None
        while process.poll() is None:
            current_progress = _read_progress(progress_file)
            if current_progress is not None and current_progress != last_progress:
                _render_progress(current_progress, progress_bar, progress_detail)
                last_progress = current_progress
            time.sleep(0.5)

        stdout, stderr = process.communicate()
        current_progress = _read_progress(progress_file)
        if current_progress is not None:
            _render_progress(current_progress, progress_bar, progress_detail)

        if result_file.exists():
            return json.loads(result_file.read_text(encoding="utf-8"))
        return {
            "ok": False,
            "outputs": [],
            "error": (
                "Worker process exited without a result. "
                f"Return code: {process.returncode}. "
                f"STDERR: {stderr.strip() or stdout.strip()}"
            ),
        }


def _read_progress(progress_file: Path) -> dict | None:
    if not progress_file.exists():
        return None
    try:
        return json.loads(progress_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _render_progress(progress: dict, progress_bar, progress_detail) -> None:
    message = str(progress.get("message") or "正在处理")
    percent = progress.get("percent")
    if isinstance(percent, int | float):
        clamped_percent = max(0.0, min(1.0, float(percent)))
        if progress_bar is not None:
            progress_bar.progress(clamped_percent, text=message)

    current = progress.get("current")
    total = progress.get("total")
    eta_seconds = progress.get("eta_seconds")
    elapsed_seconds = progress.get("elapsed_seconds")
    if current is not None and total is not None:
        detail = f"当前翻译进度：{current} / {total}"
        if isinstance(eta_seconds, int | float) and eta_seconds > 0:
            detail = f"{detail} ｜ 预计剩余时间：约 {_format_duration(eta_seconds)}"
        if isinstance(elapsed_seconds, int | float):
            detail = f"{detail} ｜ 已用时间：{_format_duration(elapsed_seconds)}"
    else:
        detail = f"当前进度：{message}"
        if isinstance(elapsed_seconds, int | float):
            detail = f"{detail} ｜ 已用时间：{_format_duration(elapsed_seconds)}"

    if progress_detail is not None:
        progress_detail.info(detail)


def _format_duration(seconds: float) -> str:
    rounded_seconds = max(0, int(round(seconds)))
    minutes, seconds_part = divmod(rounded_seconds, 60)
    hours, minutes_part = divmod(minutes, 60)
    if hours:
        return f"{hours} 小时 {minutes_part} 分 {seconds_part} 秒"
    if minutes_part:
        return f"{minutes_part} 分 {seconds_part} 秒"
    return f"{seconds_part} 秒"


def _select_media_file() -> str | None:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        file_path = filedialog.askopenfilename(
            title="选择媒体文件",
            filetypes=[
                ("媒体文件", "*.mp4 *.mkv *.mov *.avi *.wmv *.flv *.webm *.mp3 *.wav *.m4a *.aac *.flac *.ogg"),
                ("视频文件", "*.mp4 *.mkv *.mov *.avi *.wmv *.flv *.webm"),
                ("音频文件", "*.mp3 *.wav *.m4a *.aac *.flac *.ogg"),
                ("所有文件", "*.*"),
            ],
        )
        return file_path or None
    finally:
        root.destroy()


def _is_streamlit_runtime() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except ModuleNotFoundError:
        return False
    return get_script_run_ctx() is not None


if _is_streamlit_runtime():
    render_app()
elif __name__ == "__main__":
    main()
