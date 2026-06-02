# 修复文本相似度计算失真问题

## 问题
`my_video/core/optimize/optimize.py` 使用 `difflib.SequenceMatcher` 对**字符序列**计算相似度。当文本较长且存在重复词（如多个 `like`）时，算法会产生**错位匹配**，导致相似度严重失真。

示例：仅删除两个 `like`，字符级相似度从约 97% 暴跌至 **13.1%**，触发误报。

## 方案
引入 `rapidfuzz` 库，改为**按 token（词+标点）级别**计算 Levenshtein 编辑距离。

- token 通过 `str.split()` 获取，保留附着标点（如 `"games,"` 作为完整 token）
- 空格规范化已由 `split()` 隐式处理
- `rapidfuzz.distance.Levenshtein` 支持传入字符串列表（hashable sequence）

## 修改范围

### 1. `pyproject.toml`
在 `dependencies` 列表追加 `"rapidfuzz>=3.0"`

### 2. `my_video/core/optimize/optimize.py`
定位到相似度检查代码段（约第 289-295 行）：

```python
original_cleaned = re.sub(r"\s+", " ", original_text).strip()
optimized_cleaned = re.sub(r"\s+", " ", optimized_text).strip()

matcher = difflib.SequenceMatcher(None, original_cleaned, optimized_cleaned)
similarity = matcher.ratio()
```

替换为：

```python
from rapidfuzz.distance import Levenshtein

original_tokens = original_text.strip().split()
optimized_tokens = optimized_text.strip().split()

similarity = Levenshtein.normalized_similarity(original_tokens, optimized_tokens)
```

### 3. 阈值
保持现有阈值不变：
- 短句（≤10 词）：0.3
- 长句：0.7

因为 token 级 Levenshtein 更直观（相似度 ≈ 1 - 改动 token 数/总长），原有阈值语义仍然成立。

### 4. 清理
若修改后 `difflib` 在该文件中不再有其他用途，可移除顶层的 `import difflib`。

## 验证
运行 `scripts/compare_similarity_libs.py`（已存在）确认 token 级结果合理后，再跑项目测试确保无回归。
