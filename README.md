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
