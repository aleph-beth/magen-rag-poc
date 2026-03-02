from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any


class RunLogger:
    def __init__(self, runs_dir: Path) -> None:
        self.runs_dir = runs_dir
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def start(self) -> tuple[str, Path]:
        run_id = uuid.uuid4().hex[:12]
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_id, run_dir

    def write_json(self, run_dir: Path, name: str, payload: dict[str, Any]) -> None:
        (run_dir / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def write_text(self, run_dir: Path, name: str, payload: str) -> None:
        (run_dir / name).write_text(payload, encoding="utf-8")
