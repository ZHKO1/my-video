## my_video download TOML 配置

`my_video download` 会自动读取 yt-dlp 额外参数：

1. 若传入 `--config /path/to/custom.toml`，优先读取该 TOML。
2. 否则读取当前目录下默认文件 `my_video.toml`。
3. 文件不存在时会忽略并继续下载。

固定 section 为 `[download.yt-dlp.common]`：

- `output_dir`：默认输出目录（可选，支持 `~`）
- `extra_args`：追加给 `yt-dlp` 的字符串数组

```toml
[download.yt-dlp.common]
output_dir = "~/work/my-video"
extra_args = [
  "--cookies", "./cookies.txt",
  "--proxy", "socks5://192.168.71.5:20170/",
  "--remote-components", "ejs:github"
]
```

命令行 `--output` 会优先覆盖 `output_dir`。

## my_video download 行为

`my_video download` 的执行顺序：

1. 先下载 `video+audio`（`bestvideo+bestaudio/best`），并通过 `--print after_move:filepath` 获取最终视频路径。
2. 再下载字幕（`--skip-download --write-subs --convert-subs srt`），字幕目标文件名基于视频文件名（同目录，`.srt` 后缀）。

规则：

- 视频下载失败：命令直接失败。
- 字幕下载失败：仅 warning，不影响整体成功。
- 字幕路径判定：匹配与目标字幕名前缀一致的 `.srt` 文件，命中后作为字幕路径输出。
