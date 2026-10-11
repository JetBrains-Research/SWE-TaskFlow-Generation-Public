from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class Trajectory:
    source: Path
    raw: dict[str, Any]
    info: dict[str, Any]
    messages: tuple[dict[str, Any], ...]


def load_trajectory(path: Path) -> Trajectory:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid trajectory {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Trajectory {path} must contain a JSON object")
    info = value.get("info", {})
    messages = value.get("messages", ())
    if not isinstance(info, dict):
        raise ValueError(f"Trajectory {path} has invalid info")
    if not isinstance(messages, list) or not all(isinstance(message, dict) for message in messages):
        raise ValueError(f"Trajectory {path} has invalid messages")
    return Trajectory(source=path, raw=value, info=info, messages=tuple(messages))


def summarize_trajectory(trajectory: Trajectory) -> dict[str, Any]:
    config = trajectory.info.get("config", {})
    model_config = config.get("model", {}) if isinstance(config, dict) else {}
    model_stats = trajectory.info.get("model_stats", {})
    return {
        "source": str(trajectory.source),
        "message_count": len(trajectory.messages),
        "model": model_config.get("model_name") if isinstance(model_config, dict) else None,
        "exit_status": trajectory.info.get("exit_status"),
        "submission_present": bool(trajectory.info.get("submission")),
        "model_stats": model_stats if isinstance(model_stats, dict) else {},
    }
