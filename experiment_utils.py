from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


def _sanitize_token(value: Any) -> str:
    text = str(value).strip().replace(".", "p")
    text = re.sub(r"[^A-Za-z0-9_-]+", "-", text)
    text = re.sub(r"-{2,}", "-", text)
    return text.strip("-") or "na"


def make_run_slug(
    model_name: str,
    epochs: int,
    batch_size: int,
    extra_params: dict[str, Any] | None = None,
    timestamp: str | None = None,
) -> str:
    stamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [stamp, _sanitize_token(model_name), f"e{epochs}", f"bs{batch_size}"]

    if extra_params:
        for key, value in extra_params.items():
            parts.append(f"{_sanitize_token(key)}{_sanitize_token(value)}")

    return "_".join(parts)


def create_run_dir(
    output_root: str | Path,
    model_name: str,
    epochs: int,
    batch_size: int,
    extra_params: dict[str, Any] | None = None,
    timestamp: str | None = None,
) -> Path:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / make_run_slug(
        model_name=model_name,
        epochs=epochs,
        batch_size=batch_size,
        extra_params=extra_params,
        timestamp=timestamp,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_run_config(run_dir: str | Path, config: dict[str, Any]) -> Path:
    config_path = Path(run_dir) / "run_config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config_path
