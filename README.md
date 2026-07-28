# Subtitle Generator

本地视频/音频字幕生成工具。应用使用 `faster-whisper` 进行语音识别，
并在写入 `.srt` 和 `.vtt` 字幕文件前，将非中文语音翻译为简体中文。
媒体文件只在本机处理，不会上传。

## 技术说明

本项目使用 Whisper 模型进行语音识别和字幕文本提取，使用 NLLB 模型将
非中文识别结果翻译为简体中文。项目开发过程中使用了 Codex 和 DeepSeek
进行辅助编程。

## Windows 快速开始

1. 安装 Miniconda/Anaconda，或安装 Python 3.12+。
2. 下载或克隆本项目。
3. 双击 `setup.bat`。
4. 双击 `launch.bat`。

`setup.bat` 会优先使用已有的 Conda 安装。如果没有可用 Conda，它会改为
在项目目录内创建 `.venv` 本地虚拟环境。模型缓存默认放在
`.cache/huggingface`，NLLB 翻译模型放在
`models/nllb-200-distilled-600M`。如果你想使用其他缓存位置，可以自行设置
`HF_HOME`。

首次部署需要联网，并且需要为模型文件预留磁盘空间。翻译模型约 4.6 GB。
默认 Whisper 识别模型是 `small`；之后也可以在 UI 或 CLI 中选择 `medium`
以及其他 Whisper 模型。部署 `large-v3` 识别模型时，需额外预留 2.9 GB
以上磁盘空间。

默认配置使用 CPU 模式，兼容性更好但速度较慢。如果希望更快处理视频，请查看
下方“GPU 模式”部分，并按说明安装额外的 NVIDIA/CUDA 依赖。

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

准备 `large-v3` 模型时，请先确认至少有 2.9 GB 以上可用磁盘空间：

```bash
subgen-download-model --model large-v3 --device cpu --skip-translation-model
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

CPU 模式速度较慢。使用 `small` 模型时，粗略处理时长如下：

- 30 分钟视频：约 20-60 分钟
- 60 分钟视频：约 40-120 分钟
- 120 分钟视频：约 1.5-4 小时

GPU 模式通常能够大幅加速，但加速幅度取决于显卡型号、显存、CUDA 环境、
模型大小、视频内容和翻译阶段耗时，因此本项目不承诺固定的时间减少百分比。
默认一键安装以 CPU 兼容性为优先：不会自动安装 NVIDIA 驱动、CUDA 工具链、
CUDA 专用 PyTorch 或其他完整 GPU 运行环境；Whisper/NLLB 模型权重本身不区分
CPU 版和 GPU 版，是否使用 GPU 主要取决于本机依赖环境和启动参数。

从零配置 GPU 模式：

1. 确认电脑有 NVIDIA 显卡，并安装最新版 NVIDIA 驱动。
2. 在 PowerShell 或命令提示符运行 `nvidia-smi`。如果能看到显卡和驱动信息，
   说明驱动可用；如果命令不存在或报错，请先修复显卡驱动。
3. 先运行 `setup.bat` 完成默认 CPU 环境部署。
4. 激活项目环境：

```bash
conda activate subgen
```

如果你是无 Conda 安装，由 `setup.bat` 创建了 `.venv`，则运行：

```bash
.\.venv\Scripts\Activate.ps1
```

5. 打开 PyTorch 官方安装选择器，选择 Windows、Pip、Python 和与你显卡驱动匹配
   的 CUDA 版本，然后运行页面给出的安装命令。示例：

```bash
python -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

如果官方页面给出的 CUDA 版本不同，请以官方页面命令为准。

6. 安装 CUDA 12 对应的 cuBLAS 和 cuDNN 9 运行库。faster-whisper 使用的
   CTranslate2 GPU 后端需要这些 NVIDIA 动态库；Windows 用户可以按 NVIDIA
   官方说明安装 CUDA/cuDNN，或使用 faster-whisper 文档中提到的 Windows
   运行库打包方式。安装后请确保相关 DLL 所在目录已加入 `PATH`。
7. 验证 PyTorch 能看到 GPU：

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
```

8. 验证 faster-whisper GPU 能加载：

```bash
python -c "from faster_whisper import WhisperModel; WhisperModel('small', device='cuda', compute_type='float16'); print('faster-whisper CUDA OK')"
```

9. 启动 UI 后在侧边栏选择 `CUDA`，或从命令行运行：

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

## Tech Notes

This project uses Whisper models for speech recognition and subtitle text
extraction, and uses the NLLB model to translate non-Chinese recognition results
into Simplified Chinese. Codex and DeepSeek were used as programming assistants
during development.

## Quick Start on Windows

1. Install either Miniconda/Anaconda or Python 3.12+.
2. Download or clone this project.
3. Double-click `setup.bat`.
4. Double-click `launch.bat`.

`setup.bat` prefers an existing Conda installation. If Conda is not available,
it creates a project-local `.venv` instead. Model cache files default to
`.cache/huggingface`, and the NLLB translation model is placed under
`models/nllb-200-distilled-600M`. Set `HF_HOME` yourself if you prefer a
different cache location.

First-time setup requires network access and disk space for model files. The
translation model is about 4.6 GB. The default Whisper model is `small`; `medium`
and other Whisper models can be selected later from the UI or CLI. Deploying the
`large-v3` recognition model requires more than 2.9 GB of additional disk space.

The default configuration uses CPU mode, which is broadly compatible but slow.
For faster processing, see the "GPU Mode" section and install the additional
NVIDIA/CUDA dependencies described there.

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

Prepare the `large-v3` model after confirming more than 2.9 GB of available
disk space:

```bash
subgen-download-model --model large-v3 --device cpu --skip-translation-model
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

CPU mode is slow. With the `small` model, rough processing times are:

- 30-minute video: about 20-60 minutes
- 60-minute video: about 40-120 minutes
- 120-minute video: about 1.5-4 hours

GPU mode can usually speed processing up substantially, but the actual speedup
depends on the GPU, VRAM, CUDA environment, model size, video content, and
translation workload, so this project does not promise a fixed percentage
reduction. The default one-click setup prioritizes CPU compatibility: it does
not automatically install NVIDIA drivers, the CUDA toolkit, CUDA-specific
PyTorch builds, or a complete GPU runtime. Whisper/NLLB model weights are not
separate CPU and GPU variants; GPU usage mainly depends on local dependencies
and runtime options.

GPU setup from scratch:

1. Confirm that the computer has an NVIDIA GPU and install the latest NVIDIA
   driver.
2. Run `nvidia-smi` in PowerShell or Command Prompt. If it shows GPU and driver
   information, the driver is available; if it is missing or errors, fix the
   NVIDIA driver first.
3. Run `setup.bat` first to complete the default CPU-compatible setup.
4. Activate the project environment:

```bash
conda activate subgen
```

If `setup.bat` created a `.venv` because Conda was unavailable, run:

```bash
.\.venv\Scripts\Activate.ps1
```

5. Open the official PyTorch installation selector, choose Windows, Pip, Python,
   and the CUDA version supported by your driver, then run the command shown by
   that page. Example:

```bash
python -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

If the official page shows a different CUDA version, use the official command.

6. Install the CUDA 12 cuBLAS and cuDNN 9 runtime libraries. The CTranslate2 GPU
   backend used by faster-whisper needs these NVIDIA dynamic libraries. On
   Windows, install CUDA/cuDNN using NVIDIA's official instructions, or use the
   Windows runtime package option mentioned by the faster-whisper documentation.
   Make sure the DLL directory is included in `PATH`.
7. Verify that PyTorch can access the GPU:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
```

8. Verify that faster-whisper can load on GPU:

```bash
python -c "from faster_whisper import WhisperModel; WhisperModel('small', device='cuda', compute_type='float16'); print('faster-whisper CUDA OK')"
```

9. Start the UI and select `CUDA` in the sidebar, or run from the command line:

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
