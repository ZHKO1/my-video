## 安装

这个仓库使用 `uv` 管理 Python 环境与依赖。

首次使用时，先确认这两件事：

- Python 版本是 `3.13.x`
- 系统里已经安装 `ffmpeg`

`pyproject.toml` 当前限制的是 `>=3.13,<3.14`，所以 `3.12` 或 `3.14` 都不行，见 [pyproject.toml](/home/zhko1/github/my-video/pyproject.toml#L4)。转写流程会直接调用本机 `ffmpeg` 命令，见 [audio_preprocess.py](/home/zhko1/github/my-video/my_video/core/asr/audio_preprocess.py#L10)。

### 1. 创建虚拟环境

如果本机还没有 `uv`，先安装 `uv`。

然后在仓库根目录执行：

```bash
uv venv --python 3.13
```

### 2. 安装项目依赖

安装默认依赖：

```bash
uv sync
```

如果你也要安装开发依赖：

```bash
uv sync --group dev
```

默认依赖当前包含：

- `audioop-lts`（Python 3.13 下给 `pydub` 提供 `audioop` 兼容层）
- `openpyxl`
- `pandas`
- `pydub`
- `whisperx`
- `yt-dlp`

开发依赖当前包含：

- `pytest`
- `ruff`

### 3. 验证安装

激活虚拟环境后检查 CLI 是否可用：

```bash
source .venv/bin/activate
my-video --help
```

如果你不想手动激活，也可以直接用：

```bash
uv run my-video --help
```

## 可选：安装 demucs

`demucs` 不是默认依赖。原因是它的依赖声明会和 `whisperx` 使用的 `torchaudio` 版本冲突，所以这里单独提供了一个安装脚本。

如果当前环境里已经装好了 `whisperx`、`torch` 和 `torchaudio`，可以执行：

```bash
bash scripts/install_demucs_uv.sh --python .venv/bin/python
```

如果你当前已经激活了目标虚拟环境，也可以直接执行：

```bash
bash scripts/install_demucs_uv.sh
```

这个脚本会：

- 先确认目标环境里已经有 `whisperx`
- 记录安装前的 `torch` / `torchaudio` 版本
- 用 `uv pip install --no-deps` 安装 `demucs`
- 再补装 `dora-search`、`openunmix`、`lameenc`
- 最后检查 `torch` / `torchaudio` 没有被改动

这个流程只解决 `demucs` 与 `torchaudio` 的依赖冲突

## download 配置

`my_video` 会读取全局工作目录与 yt-dlp 运行参数：

1. 读取当前目录下默认文件 `my_video.toml`。
2. 文件不存在时会忽略并继续下载。

- `work_dir`：全局工作目录，可选，支持 `~`
- `download.cookie_path`：cookie 文件路径，可选
- `download.proxy`：代理地址，可选

```toml
work_dir = "~/work/my-video"

[download]
cookie_path = "./cookies.txt"
proxy = "socks5://192.168.71.5:20170/"
```

`work_dir` 为统一工作目录，所有产物都写入该目录。

## transcribe 配置

`my_video transcribe` 也读取当前目录下的 `my_video.toml`。

- `work_dir`：全局工作目录，支持 `~`
- `[transcribe].demucs`：是否启用 Demucs 人声分离，默认 `false`
- `[transcribe.whisperx].language`：WhisperX 语言，默认 `"en"`
- `[transcribe.whisperx].model`：WhisperX 模型名，默认 `"large-v3-turbo"`
- `[transcribe.whisperx].model_dir`：本地模型目录，可留空

示例：

```toml
work_dir = "~/work/my-video"

[transcribe]
demucs = true

[transcribe.whisperx]
language = "en"
model = "large-v3-turbo"
model_dir = ""
```

## download 行为

`my_video download <url>` 的执行流程：

1. 通过 yt-dlp 获取视频元信息（标题、ID 等），写入 `<workspace>/info.json`。
2. 基于视频标题创建 workspace 目录：`<work_dir>/<SanitizedTitle>/`。
3. 一次性下载视频、字幕、缩略图到 `<workspace>/origin/`：
   - **视频**：最大 1080p，格式 `(bestvideo[height<=1080]+bestaudio/best[height<=1080])`，输出为 `video.<ext>`
   - **字幕**：仅英文字幕（`en.*`），自动字幕不下载（`writeautomaticsub: False`），输出为 `subtitle.<ext>`
   - **缩略图**：输出为 `thumb.jpg`
4. 下载结果写入 `<workspace>/status.json`。

规则：

- 视频下载失败：命令直接失败
- 字幕或缩略图未找到：打印 info 提示，不影响整体成功
- 不支持播放列表（`noplaylist: True`）

## transcribe 用法

命令格式：

```bash
uv run my-video transcribe <workspace-path>
```

示例：

```bash
uv run my-video transcribe ~/work/my-video/SomeVideoTitle
```

参数说明：

- `<workspace-path>`：工作区目录路径。该目录必须存在，且包含 `status.json`（其中记录 `origin.video_path`）。通常由 `download` 命令生成。

执行流程：

1. 检查 workspace 目录是否存在。
2. 从 `<workspace>/status.json` 读取原始视频路径 `origin.video_path`。
3. 如果 `transcribe/whisperx.json` 已存在，直接返回成功（幂等）。
4. 用 `ffmpeg` 提取音频到 `transcribe/raw.mp3`。
5. 如果 `[transcribe].demucs = true`，先做 Demucs 人声分离，输出到 `transcribe/vocal.mp3`，并用人声轨做转写。
6. 用本地 WhisperX 转写音频，输出 JSON 结果到 `transcribe/whisperx.json`。
7. 扫描转写结果中的 hallucination，若发现则打印警告。
8. 从 WhisperX 结果生成 `transcribe/whisperx.srt`。

当前实现的主要输出路径（相对于 `<workspace-path>`）：

- `transcribe/whisperx.srt`：WhisperX 生成的字幕文件
- `transcribe/whisperx.json`：WhisperX 原始转写结果
- `transcribe/raw.mp3`：抽取后的原始音频
- `transcribe/vocal.mp3`：Demucs 人声轨（仅在 `demucs = true` 时生成）

注意：

- `transcribe` 接收的是 **workspace 目录**，不是视频文件本身。通常先执行 `download` 生成 workspace，再对其执行 `transcribe`。
- `demucs` 不是默认依赖；只有在 `[transcribe].demucs = true` 时才需要额外安装。
- 第一次加载 WhisperX / pyannote 模型时可能会下载模型文件。
- 当前实现不会生成词级 Excel 中间结果（该逻辑已注释掉）。
