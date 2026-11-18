# Subtitle Generator

本地视频/音频字幕生成工具。应用使用 `faster-whisper` 进行语音识别，
并在写入 `.srt` 和 `.vtt` 字幕文件前，将非中文语音翻译为简体中文。
媒体文件只在本机处理，不会上传。

## 技术说明

本项目使用 Whisper 模型进行语音识别和字幕文本提取，使用本地 GGUF 大模型
（默认 Qwen2.5-7B-Instruct，通过 llama.cpp 推理）将非中文识别结果翻译为
简体中文。项目开发过程中使用了 Codex 和 DeepSeek 进行辅助编程。

## Windows 快速开始

1. 安装 Miniconda/Anaconda，或安装 Python 3.12+。
2. 下载或克隆本项目。
3. 双击 `setup.bat`。
4. 双击 `launch.bat`。

`setup.bat` 会优先使用已有的 Conda 安装。如果没有可用 Conda，它会改为
在项目目录内创建 `.venv` 本地虚拟环境。模型缓存默认放在
`.cache/huggingface`，翻译模型放在 `models/qwen2.5-7b-instruct-q4_k_m`。
如果你想使用其他缓存位置，可以自行设置 `HF_HOME`。

首次部署需要联网，并且需要为模型文件预留磁盘空间。翻译模型（Q4_K_M）
约 4.7 GB；GGUF 不在 HuggingFace 缓存里，需要手动放进上面的目录（见下文）。
默认 Whisper 识别模型是 `small`；之后也可以在 UI 或 CLI 中选择 `medium`
以及其他 Whisper 模型。部署 `large-v3` 识别模型时，需额外预留 2.9 GB
以上磁盘空间。

默认配置会自动检测运行设备：检测到可用的 CUDA 时使用 GPU（`large-v3` + `float16`），
否则回退到 CPU（`small` + `int8`）。如果没有 GPU，处理速度会明显变慢，请查看
下方“GPU 模式”部分，并按说明安装额外的 NVIDIA/CUDA 依赖。

## 手动安装

使用 Conda：

```bash
conda env create -f environment.yml
conda activate subgen
python -m pip install -e ".[ui,translate,test]"
subgen-download-model --model small --device cpu --skip-translation-model
```

不使用 Conda：

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[ui,translate,test]"
subgen-download-model --model small --device cpu --skip-translation-model
```

llama.cpp 在 Windows 上没有可靠的 PyPI 预编译包，装完依赖后再安装项目自带的 wheel，
并把 GGUF 翻译模型放进 `models\qwen2.5-7b-instruct-q4_k_m\`：

```bash
python -m pip install --no-deps models\runtime\llama_cpp_python-*.whl
python -m pip install "diskcache>=5.6" "jinja2>=3.1"
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
subgen input.mp4 --translation-model models\qwen2.5-7b-instruct-q4_k_m
subgen input.mp4 --preset noisy
subgen input.mp4 --glossary "ヒカル=光,みえ=美惠"
subgen input.mp4 --glossary-file glossary.tsv
subgen input.mp4 --rename "希卡尔=光,美惠=美由纪"
subgen input.mp4 --batch-size 4
subgen input.mp4 --overwrite
```

默认设置：

- 输出目录：输入文件所在目录
- 字幕格式：同时生成 SRT 和 VTT
- 源语言：`auto`（日语素材建议直接指定 `ja`）
- Whisper 模型：`auto`（有可用 CUDA 时用 `large-v3`，否则用 `small`）
- 运行设备：`auto`（检测到 CUDA 就用 GPU，否则 CPU）
- 计算精度：`auto`（GPU 用 `float16`，CPU 用 `int8`）
- 识别预设：源语言为日语时自动启用"日语优化"参数（细粒度 VAD、词级时间戳、防复读）；
  背景音大、人声弱时改用 `--preset noisy`
- 后处理：过滤幻觉句与重复行、拆分过长字幕、规范中文标点
- 批量翻译：默认逐句（`--batch-size 1`）；`--batch-size N` 一次请求翻 N 行，
  行数对不上会自动回落逐句重译
- 语气词：「はい」「うん」这类单独成句的应答词直接映射成固定中文（「嗯」），
  不送进翻译模型，也不并进上一句
- 输出语言：简体中文（单语交付）

## 日语字幕质量

日语是 Whisper 的相对弱项，本项目为此做了几处针对性处理：

1. **模型**：GPU 上默认 `large-v3`，对日语的准确率远高于 `small`；显存紧张可在 UI 选
   `large-v3-turbo`。日文专用的 `kotoba-whisper-v2.0` 实测更差，不采用（见下）。
2. **VAD**：日语对话停顿短，默认 2 秒静音门槛会把相邻句子切碎，因此日语预设改用
   500 ms 静音门槛与 200 ms 语音填充。
3. **防复读**：关闭 `condition_on_previous_text`，并开启
   `hallucination_silence_threshold`，避免长视频里出现整段重复。
4. **术语提示**：用 `--hotwords` 传入人名、作品名、专业术语，比 `initial_prompt`
   更可控（后者过长时会被模型复读）。
5. **翻译**：先用本地大模型（默认 Qwen2.5-7B-Instruct）翻译，每句带上前面两轮
   原文与译文作为上下文，中文标点统一为全角；纯应答词走固定映射表，不交给模型。
   术语表命中会校验：原文出现而译文没用到指定写法的句子会带着明确要求重译一次。
6. **输出**：一条字幕对应一句话，中文单行不超过 20 字，必要时自动折行。

> 实测提醒：`avg_logprob` / `no_speech_prob` 在这类素材上几乎不区分好坏句（同一批
> 片段里，胡编的句子和正确句子的分数一样），所以本项目**不做**"低置信度自动重识别"，
> 那条路只会白费算力。识别质量的瓶颈仍在音频本身。

> 实测提醒 2：`kotoba-whisper-v2.0`（large-v3 蒸馏版，解码器只有 4 层）在同一段
> 290 秒素材上段数更少、重复更严重（出现过"普段やってるから普段はやって普段は
> やってないです"这类复读），并丢失若干句；它的 CT2 配置里 `alignment_heads`
> 仍指向第 7~25 层解码器，而实际只有 4 层，因此一开 `word_timestamps` 就会让进程
> 访问越界直接崩溃（CUDA + float16 同样会崩，只能换 int8_float16/float32）。结论：
> 不采用，日文识别继续用 `large-v3`。

### 嘈杂素材（背景音大 / 人声弱）

```bash
subgen input.mp4 --preset noisy
```

嘈杂预设把 VAD 门槛从 0.5 降到 0.30、语音填充从 200 ms 加到 300 ms，并放宽
`no_speech`/`log_prob`/`compression_ratio` 阈值。实测在 0 dB 信噪比（人声与背景音
一样响）的素材上，能多识别出约 50% 的内容。

但它救不了"人声被完全盖住"的素材：−5 dB 时 Whisper 只会吐出一句幻觉，并被本项目的
过滤器正确删除——结果是几乎空字幕。判断标准很简单：如果生成的字幕条数远少于你听到的
对话数，说明是音频层面的问题（分轨、人声太远、背景音乐压过人声），换任何识别参数都
不会有本质改善，需要先做音轨处理或人声分离。

### 人名与术语表

**不懂日语也能用。** 默认情况下翻译模型会自己把人名音译成中文，并在同一部片子里保持
一致（例如同一部片子里的 `ヒカル` 全部译成"希卡尔"、`みえ` 译成"美惠"）。如果你
不喜欢这个译名，只需要在**中文层面**改：

```bash
subgen input.mp4 --rename "希卡尔=光,美惠=美由纪"
```

`--rename` 只做中文字符串替换，不需要知道日语原词，也不需要懂日语。UI 侧边栏对应
「译名统一（中文）：旧写法=新写法」输入框。不知道模型译成了什么？先跑一遍看字幕，
或者直接用记事本把 `.srt` 里的旧名字全部替换掉，效果一样。

如果你知道日语原词（或从别处拿到了演员表/角色名），可以用术语表：

```bash
subgen input.mp4 --glossary "ヒカル=光,みえ=美惠"
subgen input.mp4 --glossary-file glossary.tsv   # 一行一条：原词=中文
```

术语表有两个作用：

- 左侧原词会**自动并入 hotwords**，让 Whisper 更可能把名字听对；
- 右侧写法会**写进翻译提示词**，要求模型按指定写法翻译（并保持前后一致）。

虚构人名在 Wikipedia/Wikidata 上通常查不到，所以这份表需要手写；一般 5–20 条就够。
现实收益主要是"前后一致"，而不是整体准确率的大幅提升。

**关于 hotwords 的取舍**：不建议从识别结果自动生成 hotwords——如果 Whisper 已经把
`ヒカル` 听成 `ヒカリ`，自动生成的提示只会把错误固化。hotwords 适合有可靠来源的场景
（片名、演员表、你从别处查到的日文写法）；只想要"名字统一"的话，用 `--rename` 就够了。

### 短句与语气词

单独成句的应答词（「はい」「うん」「えっ」「ありがとう」等）走固定映射表，不经过翻译
模型：大模型会把「はい」扩写成「好的，我明白了」，语气和长度都不对。同时：

- 带句末标点或前后有明显停顿的应答词会独立成一条字幕，不和相邻句子合并；
- 连续说出的「そう思う」不会被拆开，仍按一个句子翻译；
- 极短字幕（≤3 字）的最短展示时长从 0.8 秒提高到 1.2 秒，避免一闪而过。

如果结果仍然不理想，可以用诊断输出定位问题出在识别还是翻译：

```bash
subgen input.mkv --language ja --debug-originals --overwrite
```

它会在输出目录的 `debug/` 下生成三份**中间产物**（默认不生成，正式交付不需要）：

- `input.ja.srt`：日语原文，用于检查识别质量；
- `input.bilingual.srt`：原文 + 译文对照；
- `input.qa.jsonl`：逐条质检数据（假名占比、疑似幻觉、超长/超短等标记）。

对照之后：原文错得多 → 换更大模型或加 `--hotwords`；原文对而中文错 → 换更强的翻译模型。

完全离线运行：

```bash
subgen input.mkv --offline
```

`--offline` 只使用本地已缓存的模型，不访问网络。

## 模型下载

推荐直接让程序按设备自动选择（GPU 下载 `large-v3`，CPU 下载 `small`）：

```bash
subgen-download-model --skip-translation-model
```

也可以手动准备指定模型：

```bash
subgen-download-model --model small --device cpu --skip-translation-model
subgen-download-model --model large-v3 --device cuda --compute-type float16 --skip-translation-model
subgen-download-model --model large-v3-turbo --device cuda --compute-type float16 --skip-translation-model
```

准备 `large-v3` 模型时，请先确认至少有 2.9 GB 以上可用磁盘空间。

### 翻译模型（本地 GGUF）

翻译模型不通过 pip 或 HuggingFace 自动下载，需要把 GGUF 文件放进项目目录：

```
models\
  qwen2.5-7b-instruct-q4_k_m\
    qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf
    qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf
```

这套 Q4_K_M 权重合计约 4.7 GB，支持 2048 上下文即可运行，实测显存占用约 6.2 GB。
放好之后检查：

```bash
subgen-download-model --skip-translation-model   # 只准备识别模型
python -c "from subgen.translate import verify_translation_model; print(verify_translation_model())"
```

默认翻译模型按以下顺序解析：

1. `SUBGEN_TRANSLATION_MODEL`
2. `models/qwen2.5-7b-instruct-q4_k_m`（目录里的 `*.gguf` 或 `*-00001-of-*.gguf`）

也可以用 `--translation-model` 直接指定某个 `.gguf` 文件或目录。目录里有多个互不
相关的 GGUF 时程序会报错并要求明确指定，避免加载错模型。

## GPU 模式

程序启动时会自动检测 CUDA：可用则使用 GPU，不可用则回退 CPU，无需手动切换。
GPU 模式需要 NVIDIA 显卡、可用的 CUDA 驱动、支持 CUDA 的 faster-whisper/ctranslate2
安装，以及带 CUDA 支持的 llama.cpp wheel（翻译阶段使用）。

CPU 模式速度较慢。使用 `small` 模型时，粗略处理时长如下：

- 30 分钟视频：约 20-60 分钟
- 60 分钟视频：约 40-120 分钟
- 120 分钟视频：约 1.5-4 小时

GPU 模式通常能够大幅加速，但加速幅度取决于显卡型号、显存、CUDA 环境、
模型大小、视频内容和翻译阶段耗时，因此本项目不承诺固定的时间减少百分比。
默认一键安装以 CPU 兼容性为优先：不会自动安装 NVIDIA 驱动、CUDA 工具链或其他
完整 GPU 运行环境；Whisper 与 GGUF 模型权重本身不区分 CPU 版和 GPU 版，
是否使用 GPU 主要取决于本机依赖环境和启动参数。

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

5. 安装 CUDA 12 对应的 cuBLAS 和 cuDNN 9 运行库。faster-whisper 使用的
   CTranslate2 GPU 后端需要这些 NVIDIA 动态库；Windows 用户可以按 NVIDIA
   官方说明安装 CUDA/cuDNN，或使用 faster-whisper 文档中提到的 Windows
   运行库打包方式。安装后请确保相关 DLL 所在目录已加入 `PATH`。
6. 确认 `llama-cpp-python` 是带 CUDA 支持的版本（项目自带的 cu124 wheel 已经
   编译了 CUDA 后端）：

```bash
python -c "import llama_cpp; print(llama_cpp.__version__)"
```

7. 验证 faster-whisper GPU 能加载：

```bash
python -c "from faster_whisper import WhisperModel; WhisperModel('small', device='cuda', compute_type='float16'); print('faster-whisper CUDA OK')"
```

8. 验证 llama.cpp 能用 GPU 加载翻译模型（会打印设备信息）：

```bash
python -c "from subgen.translate import _load_llm, resolve_model_file; _load_llm(resolve_model_file(r'models\qwen2.5-7b-instruct-q4_k_m'), device='cuda', n_ctx=512); print('llama.cpp CUDA OK')"
```

9. 启动 UI 后保持"运行设备 = 自动"即可使用 GPU，也可以在侧边栏显式选择 `CUDA`，
   或从命令行运行：

```bash
subgen sample.mp4 --device cuda --compute-type float16
subgen sample.mp4 --device cuda --compute-type int8_float16
```

## 常见问题

- 如果安装脚本提示缺少 Conda，可以安装 Miniconda，或安装 Python 3.12+
  后重新运行 `setup.bat`；脚本会创建 `.venv`。
- 如果安装停在模型下载阶段，请检查是否能访问 HuggingFace。网络恢复后，
  可以运行 `subgen-download-model` 重试。
- 如果启动时报"找不到本地翻译模型"，请按上文把 GGUF 放进
  `models\qwen2.5-7b-instruct-q4_k_m\`，或用 `--translation-model` 指定路径。
- 如果某个媒体文件无法解码，请安装 FFmpeg 后重试。
- 如果选择 CUDA 后不可用，请在 UI 中切回 CPU。
- 模型缓存位置会自动选择：优先使用已含模型的项目内 `.cache/huggingface`，
  其次使用 `%USERPROFILE%\.cache\huggingface`；也可以用 `SUBGEN_HF_HOME`
  显式指定。切换 `HF_HOME` 前请先确认目标目录里已有模型，否则会重新下载。
- 开着 VPN 仍然下载缓慢或失败时：程序会自动读取 Windows 的"Internet 选项"
  代理设置并用于模型下载（Python 默认不读这个设置）。想强制直连可以设置
  `SUBGEN_NO_SYSTEM_PROXY=1`；也可以改用镜像
  `set HF_ENDPOINT=https://hf-mirror.com`，但镜像带宽通常远低于代理直连。
- 如果下载长时间停在 0 字节：`hf_xet` 通道在部分代理/VPN 下会卡死，设置
  `set HF_HUB_DISABLE_XET=1` 后重试，会回落到普通 HTTPS 下载。
- GGUF 体积较大（约 4.7 GB），建议用支持断点续传的工具下载；中断后重新下载即可，
  llama.cpp 不依赖 HuggingFace 缓存格式。

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
extraction, and a local GGUF LLM (default Qwen2.5-7B-Instruct via llama.cpp) to
translate non-Chinese recognition results into Simplified Chinese. Codex and
DeepSeek were used as programming assistants during development.

## Quick Start on Windows

1. Install either Miniconda/Anaconda or Python 3.12+.
2. Download or clone this project.
3. Double-click `setup.bat`.
4. Double-click `launch.bat`.

`setup.bat` prefers an existing Conda installation. If Conda is not available,
it creates a project-local `.venv` instead. Model cache files default to
`.cache/huggingface`, and the GGUF translation model goes under
`models/qwen2.5-7b-instruct-q4_k_m`. Set `HF_HOME` yourself if you prefer a
different cache location.

First-time setup requires network access and disk space for model files. The
Q4_K_M translation model is about 4.7 GB and is not fetched automatically; put
the GGUF files in the directory above (see "Translation model"). The default
Whisper model is `small`; `medium` and other Whisper models can be selected later
from the UI or CLI. Deploying the `large-v3` recognition model requires more than
2.9 GB of additional disk space.

The default configuration auto-detects the runtime device: CUDA when available
(`large-v3` + `float16`), otherwise CPU (`small` + `int8`). Without a GPU,
processing is noticeably slower, so see the "GPU Mode" section and install the
additional NVIDIA/CUDA dependencies described there.

## Manual Setup

Conda:

```bash
conda env create -f environment.yml
conda activate subgen
python -m pip install -e ".[ui,translate,test]"
subgen-download-model --model small --device cpu --skip-translation-model
```

Without Conda:

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[ui,translate,test]"
subgen-download-model --model small --device cpu --skip-translation-model
```

llama.cpp has no reliable prebuilt Windows wheel on PyPI. Install the bundled
wheel and place the GGUF model under `models\qwen2.5-7b-instruct-q4_k_m\`:

```bash
python -m pip install --no-deps models\runtime\llama_cpp_python-*.whl
python -m pip install "diskcache>=5.6" "jinja2>=3.1"
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
subgen input.mp4 --translation-model models\qwen2.5-7b-instruct-q4_k_m
subgen input.mp4 --rename "希卡尔=光"
subgen input.mp4 --batch-size 4
subgen input.mp4 --overwrite
```

Defaults:

- Output directory: the input file directory
- Subtitle format: both SRT and VTT
- Source language: `auto`
- Whisper model: `auto` (`large-v3` when CUDA is available, otherwise `small`)
- Device: `auto` (GPU when CUDA is available, otherwise CPU)
- Compute type: `auto` (`float16` on GPU, `int8` on CPU)
- Output language: Simplified Chinese

The translation model transliterates Japanese names into Chinese on its own and keeps
them consistent within one file. If you want a different name, you do not need any
Japanese: `--rename "希卡尔=光"` rewrites the Chinese output only. Pass `--language ja`
for Japanese media, and `--hotwords "name1, name2"` when you have a reliable source for
the original spellings. Use `--preset noisy` when background music or ambience drowns
out the voices. A glossary (`--glossary "ヒカル=光"` or `--glossary-file glossary.tsv`)
feeds the original names to Whisper as hotwords and forces the given Chinese spelling in
the output. `--debug-originals` additionally writes source-language, bilingual and QA
files under `debug/`; it is a diagnostics-only intermediate step and is off by default.
Use `--offline` to run purely from the local cache.

## Model Download

Let the app pick a model for the current device (GPU: `large-v3`, CPU: `small`):

```bash
subgen-download-model --skip-translation-model
```

Or prepare a specific model:

```bash
subgen-download-model --model small --device cpu --skip-translation-model
subgen-download-model --model large-v3 --device cuda --compute-type float16 --skip-translation-model
subgen-download-model --model large-v3-turbo --device cuda --compute-type float16 --skip-translation-model
```

Prepare the `large-v3` model after confirming more than 2.9 GB of available disk
space.

Download or repair the local translation model:

```bash
subgen-download-model --skip-translation-model   # Whisper only; GGUF is manual
python -c "from subgen.translate import verify_translation_model; print(verify_translation_model())"
```

The translation model is a local GGUF file (default: Qwen2.5-7B-Instruct Q4_K_M,
about 4.7 GB) placed under `models/qwen2.5-7b-instruct-q4_k_m`. It is resolved in
this order:

1. `SUBGEN_TRANSLATION_MODEL`
2. `models/qwen2.5-7b-instruct-q4_k_m` (single `*.gguf`, or `*-00001-of-*.gguf`)

Pass `--translation-model` to point at a specific `.gguf` file or directory.

`--batch-size N` sends N lines per request (default 1). Measured on 60 real lines,
Qwen2.5-7B shows no speed gain from batching and occasionally drifts on the merged
context, so the default stays at one line per request.

## GPU Mode

CUDA is auto-detected at startup: GPU when available, CPU otherwise. GPU mode
requires an NVIDIA GPU, a working CUDA driver, a CUDA-capable
faster-whisper/ctranslate2 installation, and a CUDA-enabled llama.cpp wheel.

CPU mode is slow. With the `small` model, rough processing times are:

- 30-minute video: about 20-60 minutes
- 60-minute video: about 40-120 minutes
- 120-minute video: about 1.5-4 hours

GPU mode can usually speed processing up substantially, but the actual speedup
depends on the GPU, VRAM, CUDA environment, model size, video content, and
translation workload, so this project does not promise a fixed percentage
reduction. The default one-click setup prioritizes CPU compatibility: it does
not automatically install NVIDIA drivers, the CUDA toolkit, or a complete GPU
runtime. Whisper and GGUF model weights are not separate CPU and GPU variants;
GPU usage mainly depends on local dependencies and runtime options.

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

5. Install the CUDA 12 cuBLAS and cuDNN 9 runtime libraries. The CTranslate2 GPU
   backend used by faster-whisper needs these NVIDIA dynamic libraries. On
   Windows, install CUDA/cuDNN using NVIDIA's official instructions, or use the
   Windows runtime package option mentioned by the faster-whisper documentation.
   Make sure the DLL directory is included in `PATH`.
6. Make sure `llama-cpp-python` is a CUDA-enabled build (the bundled cu124 wheel
   is):

```bash
python -c "import llama_cpp; print(llama_cpp.__version__)"
```

7. Verify that faster-whisper can load on GPU:

```bash
python -c "from faster_whisper import WhisperModel; WhisperModel('small', device='cuda', compute_type='float16'); print('faster-whisper CUDA OK')"
```

8. Verify that llama.cpp loads the translation model on GPU:

```bash
python -c "from subgen.translate import _load_llm, resolve_model_file; _load_llm(resolve_model_file(r'models\qwen2.5-7b-instruct-q4_k_m'), device='cuda', n_ctx=512); print('llama.cpp CUDA OK')"
```

9. Start the UI and select `CUDA` in the sidebar, or run from the command line:

```bash
subgen sample.mp4 --device cuda --compute-type float16
subgen sample.mp4 --device cuda --compute-type int8_float16
```

## Troubleshooting

- If setup says Conda is missing, either install Miniconda or install Python
  3.12+ and rerun `setup.bat`; it will create `.venv`.
- If setup stops at model download, check network access to HuggingFace and run
  `subgen-download-model` after fixing the connection.
- If startup says the local translation model is missing, put the GGUF files
  under `models/qwen2.5-7b-instruct-q4_k_m`, or pass `--translation-model`.
- If a media file cannot be decoded, install FFmpeg and try again.
- If CUDA is selected but unavailable, switch the UI back to CPU.
- The model cache location is resolved automatically: a project-local
  `.cache/huggingface` that already contains models wins, then
  `%USERPROFILE%\.cache\huggingface`; `SUBGEN_HF_HOME` overrides both. Point it at
  a directory that already has models, otherwise everything is downloaded again.

## Tests

```bash
python -m pytest
```

The test suite validates subtitle formatting, CLI behavior, UI helpers, and
model path resolution without downloading real models.
