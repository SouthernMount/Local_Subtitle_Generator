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
    from .transcribe import (
        TranscriptionError,
        cuda_device_count,
        download_model,
        resolve_default_model,
    )
    from .translate import DEFAULT_TRANSLATION_MODEL, TranslationError
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from subgen.cli import SubtitleFormat, generate_subtitles
    from subgen.transcribe import (
        TranscriptionError,
        cuda_device_count,
        download_model,
        resolve_default_model,
    )
    from subgen.translate import DEFAULT_TRANSLATION_MODEL, TranslationError


WHISPER_MODEL_OPTIONS = [
    "auto",
    "large-v3",
    "large-v3-turbo",
    "medium",
    "small",
    "base",
    "tiny",
]

TRANSCRIPTION_PRESET_UI_PARAMS = {
    None: {"beam_size": 5, "best_of": 5, "vad_filter": True},
    "ja": {"beam_size": 5, "best_of": 5, "vad_filter": True},
    "noisy": {"beam_size": 5, "best_of": 5, "vad_filter": True},
    "accurate": {"beam_size": 10, "best_of": 5, "vad_filter": True},
    "default": {"beam_size": 5, "best_of": 5, "vad_filter": True},
    "fast": {"beam_size": 1, "best_of": 1, "vad_filter": False},
}

TRANSCRIPTION_PRESET_LABELS = {
    None: "自定义（使用下方参数）",
    "ja": "日语优化 — 细 VAD / 词级时间戳 / 防复读（推荐）",
    "noisy": "嘈杂 / 人声弱 — VAD 更灵敏，背景音大时能多救回一部分内容",
    "accurate": "高精度 — 更宽 beam，约 2 倍耗时",
    "default": "平衡 — 适合英语/中文等主流语言",
    "fast": "快速 — 牺牲精度换速度",
}


def _ensure_transcription_param_state(session_state) -> None:
    if "preset_select" not in session_state:
        session_state["preset_select"] = "ja"

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
        "auto": "自动（GPU 用 float16，CPU 用 int8）",
        "default": "自动（CTranslate2 默认）",
        "float16": "float16（GPU 推荐）",
        "int8_float16": "int8_float16（GPU 省显存）",
        "int8": "int8（CPU 推荐）",
    }
    device_labels = {
        "auto": "自动（有 CUDA 就用 GPU）",
        "cuda": "CUDA（GPU）",
        "cpu": "CPU",
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
        gpu_count = cuda_device_count()
        model = st.selectbox(
            "识别模型",
            WHISPER_MODEL_OPTIONS,
            index=0,
            key="model_select",
            format_func=lambda value: (
                f"自动（按设备选择，GPU 下为 {resolve_default_model('auto')}）"
                if value == "auto"
                else value
            ),
            help="GPU 上 large-v3 的日语准确率远高于 small，显存不够可换 large-v3-turbo。",
        )
        device = st.selectbox(
            "运行设备",
            ["auto", "cuda", "cpu"],
            index=0,
            key="device_select",
            format_func=lambda value: device_labels[value],
        )
        compute_type = st.selectbox(
            "计算精度",
            ["auto", "float16", "int8_float16", "int8", "default"],
            index=0,
            key="compute_type_select",
            format_func=lambda value: compute_type_labels[value],
            help="CUDA 通常选择 auto（float16）；显存不足可试 int8_float16；CPU 用 auto（int8）。",
        )
        if gpu_count > 0:
            st.caption(f"检测到 {gpu_count} 个 CUDA 设备，默认使用 GPU 推理。")
        else:
            st.caption("未检测到可用的 CUDA 设备，将使用 CPU 推理（large-v3 会很慢）。")
        if device == "cuda":
            st.info("GPU 模式需要 NVIDIA 显卡、可用 CUDA 驱动，以及支持 CUDA 的 PyTorch/ctranslate2。")
        language = st.selectbox(
            "源语言",
            ["auto", "ja", "zh", "en", "ko"],
            index=0,
            key="language_select",
            format_func=lambda value: f"{language_labels[value]} ({value})",
            help="日语素材建议直接选 ja：自动检测偶尔会被片头曲/音乐带偏。",
        )
        subtitle_format = st.radio(
            "字幕格式",
            [SubtitleFormat.both.value, SubtitleFormat.srt.value, SubtitleFormat.vtt.value],
            horizontal=True,
            key="subtitle_format_radio",
            format_func=lambda value: format_labels[value],
        )
        translation_model = st.text_input(
            "中文翻译模型（本地 GGUF）",
            value=DEFAULT_TRANSLATION_MODEL,
            key="translation_model_text",
            help=(
                "非中文语音会先识别为源语言文本，再用这个本地大模型翻译成中文。"
                "可以填 .gguf 文件，也可以填包含分片的目录。"
            ),
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
                    step=1,
                    key="best_of_input",
                    help="temperature > 0 时的候选数。只在触发重试时才有效。",
                    on_change=_mark_transcription_preset_custom,
                    args=(st.session_state,),
                )
            vad_filter = st.checkbox(
                "启用 VAD 语音活动检测",
                key="vad_filter_checkbox",
                help="用 Silero VAD 模型先过滤静音段，大幅减少 Whisper 的幻觉输出。强烈推荐开启。",
                on_change=_mark_transcription_preset_custom,
                args=(st.session_state,),
            )
            initial_prompt = st.text_area(
                "initial_prompt（引导文本）",
                value="",
                key="initial_prompt_text",
                placeholder="一般留空。请优先使用下方「hotwords」。",
                help="Whisper 会把它当作上文来引导识别；过长或不自然的提示词会诱发复读和幻觉。",
                on_change=_mark_transcription_preset_custom,
                args=(st.session_state,),
            )
            hotwords = st.text_input(
                "hotwords（人名/术语提示）",
                value="",
                key="hotwords_text",
                placeholder="登场人物名, 作品名, 专业术语（逗号分隔）",
                help="比 initial_prompt 更可控：只作为提示注入，用来稳定专有名词的写法。",
            )
            glossary_text = st.text_area(
                "术语表（一行一条：素材写法=中文）",
                value="",
                key="glossary_text",
                height=96,
                placeholder="ヒカル=光\nみえ=美惠\n# 以 # 开头的行会被忽略",
                help=(
                    "左侧原词会自动加进 hotwords 改善识别，右侧写法会强制统一译文里的译名。"
                    "虚构人名在百科上查不到，需要手写；一般 5–20 条即可。"
                ),
            )
            rename_text = st.text_input(
                "译名统一（中文）：旧写法=新写法",
                value="",
                key="rename_text",
                placeholder="希卡尔=光, 美惠=美由纪",
                help=(
                    "只改中文，不需要懂日语：先正常跑一遍，看到译文里的名字写法，"
                    "再把「旧写法=想要的写法」填进来重跑即可。"
                ),
            )
            batch_size = st.number_input(
                "每批翻译行数",
                min_value=1,
                max_value=8,
                value=1,
                step=1,
                key="batch_size_number",
                help=(
                    "一次请求翻译多行：相邻台词一起送进去，跨句指代更稳、速度更快；"
                    "行数对不上时会自动回落成逐句重译。"
                ),
            )
            debug_originals = st.checkbox(
                "输出诊断文件（原文 SRT / 双语对照 / QA 报告）",
                value=False,
                key="debug_originals_checkbox",
                help=(
                    "仅用于调参：在输出目录的 debug/ 下额外生成日语原文、双语对照和逐条质检数据。"
                    "正式交付时保持关闭。"
                ),
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
        if st.button("准备识别模型 / 检查翻译模型", use_container_width=True, key="download_models_button"):
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
        if hotwords.strip():
            worker_args.extend(["--hotwords", hotwords.strip()])
        if glossary_text.strip():
            worker_args.extend(["--glossary", glossary_text.strip()])
        if rename_text.strip():
            worker_args.extend(["--rename", rename_text.strip()])
        if batch_size > 1:
            worker_args.extend(["--batch-size", str(int(batch_size))])
        if debug_originals:
            worker_args.append("--debug-originals")

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
        "Glossary file does not exist:": "术语表文件不存在：",
        "Media file path is required.": "请填写媒体文件路径。",
        "Missing dependency 'faster-whisper'.": "缺少依赖 faster-whisper。",
        "Transcription failed.": "语音识别失败。",
        "Model download or load failed.": "识别模型下载或加载失败。",
        "Check your network, model name, local cache, and selected device.": (
            "请检查网络、模型名称、本地缓存和运行设备。"
        ),
        "Check your network, model name, and local cache.": (
            "请检查网络、模型名称和本地缓存。"
        ),
        "Check that the media file is readable, the model can be downloaded or loaded locally, and the selected device is available.": (
            "请确认媒体文件可读取，识别模型可下载或已存在于本地缓存，并且所选运行设备可用。"
        ),
        "Cause: ": "原因：",
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
