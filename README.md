# Subtitle Generator

本地视频/音频字幕生成工具。应用使用 `faster-whisper` 进行语音识别，
并在写入 `.srt` 和 `.vtt` 字幕文件前，将非中文语音翻译为简体中文。
媒体文件只在本机处理，不会上传。

## Windows 快速开始

1. 安装 Miniconda/Anaconda，或安装 Python 3.12+。
2. 下载或克隆本项目。
3. 双击 `setup.bat`。
4. 双击 `launch.bat` 或 `启动字幕生成器.bat`。

`setup.bat` 会优先使用已有的 Conda 安装。如果没有可用 Conda，它会改为
在项目目录内创建 `.venv` 本地虚拟环境。模型缓存默认放在
`.cache/huggingface`，NLLB 翻译模型放在
`models/nllb-200-distilled-600M`。如果你想使用其他缓存位置，可以自行设置
`HF_HOME`。

首次部署需要联网，并且需要为模型文件预留磁盘空间。翻译模型约 2.4 GB。
默认 Whisper 识别模型是 `small`；之后也可以在 UI 或 CLI 中选择 `medium`
以及其他 Whisper 模型。

## 手动安装

使用 Conda：

```bash
conda env create -f environment.yml
conda activate subgen
python -m pip install -e ".[ui,translate,test]"
python scripts/download_models.py
subgen-download-model --model small --device cpu --skip-translation-model
```

不使用 Conda：

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[ui,translate,test]"
python scripts/download_models.py
subgen-download-model --model small --device cpu --skip-translation-model
```

如果依赖下载很慢或失败，可以换用可访问的 PyPI 镜像重试：

```bash
python -m pip install --prefer-binary -i https://pypi.tuna.tsinghua.edu.cn/simple -e ".[ui,translate,test]"
```

## 使用方法

启动可视化界面：

```bash
subgen-ui --port 8501
```

从命令行生成字幕：

```bash
subgen input.mp4
subgen input.mp4 --output-dir subtitles
subgen input.mp4 --format srt
subgen input.mp4 --format vtt
subgen input.mp4 --format both
subgen input.mp4 --language auto
subgen input.mp4 --model small
subgen input.mp4 --model medium
subgen input.mp4 --device cpu
subgen input.mp4 --translation-model models\nllb-200-distilled-600M
subgen input.mp4 --overwrite
```

默认设置：

- 输出目录：输入文件所在目录
- 字幕格式：同时生成 SRT 和 VTT
- 源语言：`auto`
- Whisper 模型：`small`
- 运行设备：`cpu`
- 计算精度：`default`
- 输出语言：简体中文

## 模型下载

准备默认的低资源占用模型：

```bash
subgen-download-model --model small --device cpu --skip-translation-model
```

准备质量更高的模型：

```bash
subgen-download-model --model medium --device cpu --skip-translation-model
```

下载或修复本地翻译模型：

```bash
python scripts/download_models.py
python scripts/download_models.py --force
```

默认翻译模型按以下顺序解析：

1. `SUBGEN_TRANSLATION_MODEL`
2. `models/nllb-200-distilled-600M`
3. `facebook/nllb-200-distilled-600M`

## GPU 模式

默认使用 CPU 模式，以获得更好的兼容性。GPU 模式需要 NVIDIA 显卡、可用的
CUDA 驱动、支持 CUDA 的 PyTorch，以及支持 CUDA 的
faster-whisper/ctranslate2 安装。

```bash
subgen sample.mp4 --device cuda --compute-type float16
subgen sample.mp4 --device cuda --compute-type int8_float16
```

使用 GPU 模式前，请先安装与你的 CUDA 版本匹配的 PyTorch。

## 常见问题

- 如果安装脚本提示缺少 Conda，可以安装 Miniconda，或安装 Python 3.12+
  后重新运行 `setup.bat`；脚本会创建 `.venv`。
- 如果安装停在模型下载阶段，请检查是否能访问 HuggingFace。网络恢复后，
  可以运行 `python scripts/download_models.py --force`。
- 如果某个媒体文件无法解码，请安装 FFmpeg 后重试。
- 如果选择 CUDA 后不可用，请在 UI 中切回 CPU。

## 测试

```bash
python -m pytest
```

测试套件会验证字幕格式、CLI 行为、UI 辅助逻辑和模型路径解析，不会下载真实模型。

---

# English Version

# Subtitle Generator

Local subtitle generation for video and audio files. The app uses
`faster-whisper` for speech recognition and translates non-Chinese speech to
Simplified Chinese before writing `.srt` and `.vtt` files. Media files are
processed locally and are not uploaded.

## Quick Start on Windows

1. Install either Miniconda/Anaconda or Python 3.12+.
2. Download or clone this project.
3. Double-click `setup.bat`.
4. Double-click `launch.bat` or `启动字幕生成器.bat`.

`setup.bat` prefers an existing Conda installation. If Conda is not available,
it creates a project-local `.venv` instead. Model cache files default to
`.cache/huggingface`, and the NLLB translation model is placed under
`models/nllb-200-distilled-600M`. Set `HF_HOME` yourself if you prefer a
different cache location.

First-time setup requires network access and disk space for model files. The
translation model is about 2.4 GB. The default Whisper model is `small`; `medium`
and other Whisper models can be selected later from the UI or CLI.

## Manual Setup

Conda:

```bash
conda env create -f environment.yml
conda activate subgen
python -m pip install -e ".[ui,translate,test]"
python scripts/download_models.py
subgen-download-model --model small --device cpu --skip-translation-model
```

Without Conda:

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[ui,translate,test]"
python scripts/download_models.py
subgen-download-model --model small --device cpu --skip-translation-model
```

If dependency downloads are slow or fail, retry with a reachable PyPI mirror:

```bash
python -m pip install --prefer-binary -i https://pypi.tuna.tsinghua.edu.cn/simple -e ".[ui,translate,test]"
```

## Usage

Start the UI:

```bash
subgen-ui --port 8501
```

Generate subtitles from the command line:

```bash
subgen input.mp4
subgen input.mp4 --output-dir subtitles
subgen input.mp4 --format srt
subgen input.mp4 --format vtt
subgen input.mp4 --format both
subgen input.mp4 --language auto
subgen input.mp4 --model small
subgen input.mp4 --model medium
subgen input.mp4 --device cpu
subgen input.mp4 --translation-model models\nllb-200-distilled-600M
subgen input.mp4 --overwrite
```

Defaults:

- Output directory: the input file directory
- Subtitle format: both SRT and VTT
- Source language: `auto`
- Whisper model: `small`
- Device: `cpu`
- Compute type: `default`
- Output language: Simplified Chinese

## Model Download

Prepare the default low-resource model:

```bash
subgen-download-model --model small --device cpu --skip-translation-model
```

Prepare a higher-quality model:

```bash
subgen-download-model --model medium --device cpu --skip-translation-model
```

Download or repair the local translation model:

```bash
python scripts/download_models.py
python scripts/download_models.py --force
```

The default translation model is resolved in this order:

1. `SUBGEN_TRANSLATION_MODEL`
2. `models/nllb-200-distilled-600M`
3. `facebook/nllb-200-distilled-600M`

## GPU Mode

CPU mode is the default and is recommended for broad compatibility. GPU mode
requires an NVIDIA GPU, a working CUDA driver, CUDA-capable PyTorch, and a
CUDA-capable faster-whisper/ctranslate2 installation.

```bash
subgen sample.mp4 --device cuda --compute-type float16
subgen sample.mp4 --device cuda --compute-type int8_float16
```

Install the PyTorch build that matches your CUDA version before using GPU mode.

## Troubleshooting

- If setup says Conda is missing, either install Miniconda or install Python
  3.12+ and rerun `setup.bat`; it will create `.venv`.
- If setup stops at model download, check network access to HuggingFace and run
  `python scripts/download_models.py --force` after fixing the connection.
- If a media file cannot be decoded, install FFmpeg and try again.
- If CUDA is selected but unavailable, switch the UI back to CPU.

## Tests

```bash
python -m pytest
```

The test suite validates subtitle formatting, CLI behavior, UI helpers, and
model path resolution without downloading real models.
