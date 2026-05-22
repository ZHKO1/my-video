from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_json(path: str | Path) -> dict[str, Any] | None:
    file_path = Path(path)
    if not file_path.is_file():
        return None
    return json.loads(file_path.read_text(encoding="utf-8"))
