# Download + Workspace Split

## Summary
保留 `DownloadResult` 不变，但把“在 workspace 内创建 `origin/` + 直接下载到 `origin/`”统一放回 `my_video/core/downloader.py`。  
新增 `my_video/core/workspace.py`，只负责 workspace 名称、workspace 根目录创建、以及 `info.json` 的读写。

最终职责边界：

- `workspace.py`
  - 根据标题生成合法 workspace 名
  - 在 `work_dir` 下创建 workspace 根目录
  - 管理 `info.json`
- `downloader.py`
  - 获取标题
  - 在给定 workspace 下创建 `origin/`
  - 直接把视频/字幕/缩略图下载到 `origin/`
  - 返回最终 `DownloadResult`

## Key Changes
- `my_video/core/downloader.py`
  - 保留 `DownloadResult` 不变：
    - `video_path: str | None`
    - `subtitle_path: str | None`
    - `thumbnail_path: str | None`
  - 调整 `DownloadRequest`，让它包含下载和最终落盘所需信息：
    - `url: str`
    - `workspace_path: str | Path`
    - `cookie_path: str | Path | None`
    - `proxy: str | None`
  - 新增标题获取函数，供编排层先取标题：
    - 建议接口：`fetch_video_title(url, cookie_path=None, proxy=None) -> str`
  - `download(request)` 负责：
    - 在 `request.workspace_path` 下创建 `origin/`
    - 调用 yt-dlp 直接下载到 `origin/`
    - 固定命名为：
      - `video.<ext>`
      - `subtitle.<ext>`
      - `thumb.<ext>`
    - 返回最终绝对路径组成的 `DownloadResult`
    - 保留打印：成功路径、缺失字幕/缩略图提示、失败打印
- 新增 `my_video/core/workspace.py`
  - 只负责 workspace 和 `info.json`
  - 建议接口：
    - `sanitize_workspace_name(title: str) -> str`
    - `create_workspace(work_dir: str | Path, title: str) -> WorkspaceInfo`
    - `write_workspace_info(...) -> None`
  - `WorkspaceInfo` 至少包含：
    - `workspace_path: str`
    - `info_path: str`
    - `title: str`
- `my_video/cli/commands/download.py`
  - 负责编排，不处理文件搬运
  - 流程固定为：
    1. 读取 `work_dir`、`download.cookie_path`、`download.proxy`
    2. 调用 `fetch_video_title(...)`
    3. 标题缺失直接失败
    4. 调用 `workspace.create_workspace(...)`
    5. 先写初始 `info.json`
    6. 调用 `downloader.download(...)`
    7. 用返回结果更新 `info.json`
    8. 根据是否有 `video_path` 返回退出码

## info.json Shape
固定为：

```json
{
  "stage": "download",
  "status": "in_progress",
  "failed_reason": null,
  "content": {
    "url": "...",
    "title": "...",
    "video_path": "...",
    "subtitle_path": "...",
    "thumbnail_path": "..."
  }
}
```

固定规则：

- `stage` 固定为 `"download"`
- `status` 只允许 `"in_progress"`、`"success"` 或 `"failed"`
- `failed_reason`
  - 进行中时为 `null`
  - 成功时为 `null`
  - 失败时为可读错误信息
- `content.url` 为输入 URL
- `content.title` 为视频标题
- `content.video_path`、`content.subtitle_path`、`content.thumbnail_path` 为 workspace 中最终文件的绝对路径
- 字幕或缩略图缺失时字段写 `null`

## Workflow Details
- 标题获取
  - 在下载前先取标题
  - 标题缺失、为空、或清洗后为空时直接报错
- workspace 创建
  - 使用清洗后的标题创建 `work_dir/<sanitized_title>/`
  - 若目录已存在，直接报错
  - 这一步只创建 workspace 根目录，不创建 `origin/`
- 下载与最终落盘
  - `downloader.download()` 在 workspace 下创建 `origin/`
  - yt-dlp 直接下载到 `origin/`，命名保持固定基名
- 状态规则
  - workspace 创建后、下载完成前：`in_progress`
  - 视频存在：`success`
  - 字幕/缩略图缺失：仍为 `success`
  - 标题缺失、目录冲突、下载异常、`origin/` 创建失败、视频缺失：`failed`

## Failure Model
- 标题缺失
  - 直接失败
  - 不创建 workspace
  - 不写 `info.json`
- 目录冲突
  - 直接失败
  - 不覆盖已有目录
  - 不写新 `info.json`
- workspace 创建后失败
  - 只要 workspace 根目录已经创建，就必须写 `info.json`
  - 初始先写一份进行中状态
  - 下载失败、视频缺失、`origin/` 创建失败都写失败态：
    - `stage: "download"`
    - `status: "failed"`
    - `failed_reason: <错误信息>`
    - `content.url`
    - `content.title`
    - 其余路径字段按已知值写入，未知写 `null`

## Test Plan
- downloader
  - `fetch_video_title()` 能返回标题
  - 标题获取时能正确带上 `cookiefile`、`proxy`
  - `download()` 会在 workspace 下创建 `origin/`
  - `download()` 会直接把产物写到 `origin/`
  - `download()` 返回的仍是原样 `DownloadResult`
  - 字幕/缩略图缺失时仍成功
  - 视频缺失时失败
- workspace
  - 标题清洗正确
  - 非法字符标题可转成合法目录名
  - 同名 workspace 已存在时直接失败
  - `info.json` 的进行中、成功态和失败态都能正确写出
  - `content` 直接包含 `video_path`、`subtitle_path`、`thumbnail_path`
- CLI 编排
  - `download` 命令仍只解析 `url`
  - 标题缺失时返回 `EXIT.RUNTIME_ERROR`
  - workspace 创建后下载成功时返回 `EXIT.SUCCESS`
  - workspace 创建后下载失败时返回 `EXIT.RUNTIME_ERROR`

## Assumptions
- 这次只改下载链路，不调整 `transcribe/subtitle/synthesize`
- `work_dir` 仍来自根级配置，表示 workspace 根目录
- `download.cookie_path` 和 `download.proxy` 仍来自 `[download]`
- `origin/` 的创建和直接落盘属于下载产物落盘的一部分，因此放在 `downloader.py`
