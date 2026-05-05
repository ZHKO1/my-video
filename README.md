## 安装

这个仓库使用 `uv` 管理 Python 环境与依赖。

首次使用时，先确认这两件事：

- Python 版本是 `3.13.x`
- 系统里已经安装 `ffmpeg`

`pyproject.toml` 当前限制的是 `>=3.13,<3.14`，所以 `3.12` 或 `3.14` 都不行，见 [pyproject.toml](/home/zhko1/github/my-video/pyproject.toml#L4)。转写流程会直接调用本机 `ffmpeg` 命令，见 [audio_preprocess.py](/home/zhko1/github/my-video/my_video/core/asr_backend/audio_preprocess.py#L10)。

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

`my_video` 会读取全局工作目录与 `yt-dlp` 额外参数：

1. 读取当前目录下默认文件 `my_video.toml`。
2. 文件不存在时会忽略并继续下载。

- `work_dir`：全局工作目录，可选，支持 `~`
- `[download.yt-dlp.common].extra_args`：追加给 `yt-dlp` 的字符串数组

```toml
work_dir = "~/work/my-video"

[download.yt-dlp.common]
extra_args = [
  "--cookies", "./cookies.txt",
  "--proxy", "socks5://192.168.71.5:20170/",
  "--remote-components", "ejs:github"
]
```

`work_dir` 为统一工作目录，所有产物都写入该目录。

## download 行为

`my_video download` 的执行顺序：

1. 先下载 `video+audio`，格式选择 `bestvideo+bestaudio/best`，并通过 `--print after_move:filepath` 获取最终视频路径。
2. 再下载字幕，使用 `--skip-download --write-subs --convert-subs srt`，字幕目标文件名基于视频文件名生成。

规则：

- 视频下载失败：命令直接失败
- 字幕下载失败：只打印 warning，不影响整体成功
- 字幕路径判定：匹配与目标字幕名前缀一致的 `.srt` 文件，命中后作为字幕路径输出
