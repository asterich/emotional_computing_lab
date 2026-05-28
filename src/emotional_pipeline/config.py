from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


LABELS = ["anger", "disgust", "sadness", "joy", "neutral", "surprise", "fear"]
DEFAULT_MODEL = "qwen-vl-plus"


@dataclass(frozen=True)
class Paths:
    root: Path
    data_raw: Path
    data_processed: Path
    frames: Path
    prompts: Path
    results: Path
    raw_outputs: Path
    parsed_outputs: Path
    final_for_b: Path
    logs: Path

    @classmethod
    def from_root(cls, root: Path) -> "Paths":
        results = root / "results"
        return cls(
            root=root,
            data_raw=root / "data" / "raw",
            data_processed=root / "data" / "processed",
            frames=root / "frames" / "extracted_frames",
            prompts=root / "prompts",
            results=results,
            raw_outputs=results / "raw_outputs",
            parsed_outputs=results / "parsed_outputs",
            final_for_b=results / "final_for_B",
            logs=root / "logs",
        )

    def ensure(self) -> None:
        for path in [
            self.data_raw,
            self.data_processed,
            self.frames,
            self.prompts,
            self.raw_outputs / "batch_inputs",
            self.raw_outputs / "closed_set",
            self.raw_outputs / "open_vocab",
            self.raw_outputs / "perturbation",
            self.parsed_outputs,
            self.final_for_b,
            self.logs,
        ]:
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ApiConfig:
    api_key: Optional[str]
    base_url: str
    model: str

    @property
    def enabled(self) -> bool:
        return bool(self.api_key and self.base_url)


def load_envrc(root: Path) -> Dict[str, str]:
    """Load simple `export KEY=value` entries without executing shell code."""
    envrc = root / ".envrc"
    values: Dict[str, str] = {}
    if not envrc.exists():
        return values
    for raw_line in envrc.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or not line.startswith("export "):
            continue
        key_value = line[len("export ") :]
        if "=" not in key_value:
            continue
        key, value = key_value.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def get_env(root: Path, names: Iterable[str], default: Optional[str] = None) -> Optional[str]:
    envrc = load_envrc(root)
    for name in names:
        if os.environ.get(name):
            return os.environ[name]
        if envrc.get(name):
            return envrc[name]
    return default


def api_config(root: Path) -> ApiConfig:
    base_url = get_env(
        root,
        ["QWEN_API_BASE_URL", "DASHSCOPE_BASE_URL"],
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    return ApiConfig(
        api_key=get_env(root, ["QWEN_API_KEY", "DASHSCOPE_API_KEY"]),
        base_url=(base_url or "").rstrip("/"),
        model=get_env(root, ["QWEN_MODEL"], DEFAULT_MODEL) or DEFAULT_MODEL,
    )


def now_run_id() -> str:
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d_%H%M%S")


def relative_or_empty(root: Path, paths: List[Path]) -> str:
    values = []
    for path in paths:
        try:
            values.append(str(path.relative_to(root)))
        except ValueError:
            values.append(str(path))
    return "|".join(values)
