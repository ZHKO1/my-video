# my-video

命令行视频字幕流水线：

1. `download`：用 `yt-dlp` 下载视频、原始字幕、缩略图
2. `transcribe`：抽音频，用 WhisperX 转写
3. `subtitle`：按句分组，LLM 优化、拆行、翻译
4. `synthesize`：把双语字幕烧录回视频

当前 CLI 实际可用命令：

```bash
uv run my-video --help
```

## 环境要求

- Python `3.13.x`
- 本机已安装 `ffmpeg`
- 能运行 WhisperX / PyTorch
- 已设置 LLM 环境变量：
  - `MY_VIDEO_OPENAI_BASE_URL`
  - `MY_VIDEO_OPENAI_API_KEY`

`pyproject.toml` 当前限制为 `>=3.13,<3.14`。

## 安装

这个仓库使用 `uv` 管理环境和依赖。

### 创建虚拟环境

```bash
uv venv --python 3.13
```

### 安装默认依赖

```bash
uv sync
```

默认依赖已经包含完整流水线所需的主要包，包括：

- `openai`
- `json-repair`
- `diskcache`
- `torch`
- `whisperx`
- `demucs`
- `yt-dlp`
- `opencv-python`
- `pydub`

### 安装开发依赖

```bash
uv sync --group dev
```

### 验证 CLI

```bash
uv run my-video --help
```

## 配置

程序只读取当前工作目录下的 `my_video.toml`。

最小示例：

```toml
work_dir = "~/work/my-video"

[download]
cookie_path = "./cookies.txt"
proxy = "socks5://127.0.0.1:7890"

[transcribe]
demucs = true

[transcribe.whisperx]
language = "en"
model = "large-v3-turbo"
model_dir = ""

[subtitle]
thread_num = 4
batch_size = 20
need_reflect = true
max_word_count_cjk = 16
max_word_count_english = 18

[llm]
model = "deepseek-v4-pro"

[synthesize]
ffmpeg_gpu = false
```

当前代码实际读取的配置项如下。

### 全局

- `work_dir`：workspace 根目录，默认当前目录

### download

- `download.cookie_path`
- `download.proxy`

### transcribe

- `transcribe.demucs`：是否做人声分离，默认 `true`
- `transcribe.whisperx.language`：默认 `"en"`，也可设为 `"auto"`
- `transcribe.whisperx.model`：默认 `"large-v3-turbo"`
- `transcribe.whisperx.model_dir`：本地模型目录，可空

### subtitle

- `subtitle.thread_num`：默认 `4`
- `subtitle.batch_size`：默认 `20`
- `subtitle.need_reflect`：默认 `true`
- `subtitle.max_word_count_cjk`：默认 `16`
- `subtitle.max_word_count_english`：默认 `18`
- `llm.model`：默认 `"deepseek-v4-pro"`

说明：

- 当前翻译目标语言在代码里固定为 `简体中文`
- LLM 调用使用环境变量，不从 `my_video.toml` 读取 key/base_url

### synthesize

- `synthesize.ffmpeg_gpu`：默认 `false`

## 命令用法

### 1. download

```bash
uv run my-video download <url>
```

行为：

- 先抓取视频信息并写入 `info.json`
- 按视频标题创建 workspace
- 下载视频到 `origin/video.*`
- 下载英文字幕到 `origin/subtitle.srt`
- 下载缩略图到 `origin/thumb.jpg`
- 写入和更新 `status.json`

当前下载实现的关键点：

- `noplaylist = True`
- 视频格式优先限制在 `1080p`
- `writesubtitles = True`
- `writeautomaticsub = True`
- `subtitleslangs = ["en.*"]`
- 字幕会通过 yt-dlp postprocessor 转成 `srt`

### 2. transcribe

```bash
uv run my-video transcribe <workspace-path>
```

输入要求：

- workspace 必须已存在
- `status.json` 里必须有 `origin.video_path`

执行流程：

1. 抽取音频到 `transcribe/raw.mp3`
2. 如果 `transcribe.demucs = true`，做人声分离，输出 `transcribe/vocal.mp3`
3. 用 WhisperX 转写到 `transcribe/whisperx.json`
4. 扫描 hallucination 并打印告警
5. 导出 `transcribe/whisperx.srt`

补充说明：

- 如果 `transcribe/whisperx.json` 已存在，命令会直接成功返回
- WhisperX 会自动选择 `cuda` 或 `cpu`
- 代码会先探测 HuggingFace 官方站和镜像站，选择响应更快的下载端点

### 3. subtitle

```bash
uv run my-video subtitle <workspace-path>
```

输入要求：

- workspace 必须已存在
- `transcribe/whisperx.json` 必须存在

执行流程：

1. 从 WhisperX 词级结果构造 `SubtitleSegments`
2. 按句末标点组装成 `SubtitleSentence`
3. 如果 workspace 里有原始字幕，作为优化阶段参考文本
4. LLM 优化句子内容
5. 基于优化后的结果重新分句
6. LLM 拆成字幕行
7. LLM 翻译为简体中文
8. 产出双语 SRT

当前产物：

- `subtitle/origin.txt`：原始句子文本
- `subtitle/optimized.txt`：优化后的句子文本/优化日志
- `subtitle/split.txt`：拆行后的文本视图
- `subtitle/src.srt`：源语言字幕
- `subtitle/trans.srt`：中文字幕

补充说明：

- `SubtitleLine` 当前索引语义是：
  - `sentence_index`：对应 `SubtitleSentence.index`
  - `sentence_splited_line_index`：同一句拆分后的行号，从 `0` 开始
  - `line_index`：全局行号，从 `0` 开始
- 翻译缓存和回填现在都基于 `line_index`
- 命令还会在仓库根目录写一个调试文件 `tmp.json`

### 4. synthesize

```bash
uv run my-video synthesize <workspace-path>
```

输入要求：

- workspace 必须已存在
- `status.json` 里必须有 `origin.video_path`
- `subtitle/src.srt` 和 `subtitle/trans.srt` 必须存在

执行流程：

1. 读取原视频分辨率
2. 用 FFmpeg `subtitles` 滤镜依次叠加源字幕和译文字幕
3. 需要时启用 `h264_nvenc`
4. 生成 `output.mp4`
5. 根据 `info.json` 生成 `output.md`

当前样式规则：

- 字体文件：`my_video/assets/SourceHanSansSC-Regular.otf`
- 源字幕：白字、黑描边、阴影
- 译文字幕：青色、黑描边、半透明背景

## 完整流程

```bash
uv run my-video download <url>
uv run my-video transcribe ~/work/my-video/<VideoTitle>
uv run my-video subtitle ~/work/my-video/<VideoTitle>
uv run my-video synthesize ~/work/my-video/<VideoTitle>
```

## Workspace 结构

典型目录结构如下：

```text
<workspace>/
  info.json
  status.json
  origin/
    video.<ext>
    subtitle.srt
    thumb.jpg
  transcribe/
    raw.mp3
    vocal.mp3
    whisperx.json
    whisperx.srt
  subtitle/
    origin.txt
    optimized.txt
    split.txt
    src.srt
    trans.srt
  output.mp4
  output.md
```

`status.json` 会记录：

- `stage`
- `status`
- `failed_reason`
- `origin.video_path`
- `origin.subtitle_path`
- `origin.thumbnail_path`
- `last_time`

## 开发

### Ruff

```bash
uv run ruff check --fix .
uv run ruff format .
```

### pre-commit

```bash
uv run pre-commit install
```

### 测试

轻量测试：

```bash
uv run --no-default-groups --group test pytest
```

完整测试：

```bash
uv run pytest
```

### CI

仓库当前有一个 GitHub Actions 工作流：

- `.github/workflows/ci.yml`

它会执行：

```bash
uv run --no-default-groups --group dev ruff check .
uv run --no-default-groups --group dev ruff format --check .
uv run --no-default-groups --group test pytest
```