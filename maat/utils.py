"""Shared utilities — logging, file I/O, result formatting."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console

from maat.config import LOG_DIR, RESULTS_DIR

console = Console()


def timestamp() -> str:
    """Return ISO timestamp string."""
    return datetime.now(timezone.utc).isoformat()


def make_log_path(domain: str, model_name: str) -> Path:
    """Create a log file path with domain, model, and timestamp."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = model_name.replace("/", "_").replace(":", "_")
    return LOG_DIR / f"{domain}_{safe_model}_{ts}.txt"


def make_result_path(domain: str, model_name: str) -> Path:
    """Create a result JSON file path."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = model_name.replace("/", "_").replace(":", "_")
    return RESULTS_DIR / f"{domain}_{safe_model}_{ts}.json"


def save_result(path: Path, data: dict) -> None:
    """Save structured result as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    console.print(f"[green]Result saved:[/green] {path}")


def log_round(log_path: Path, role: str, content: str, round_num: int | None = None) -> None:
    """Append a round entry to the log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    prefix = f"[Round {round_num}] " if round_num else ""
    entry = f"\n{'='*60}\n{prefix}{role.upper()} [{timestamp()}]\n{'='*60}\n{content}\n"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(entry)
    # Also print to console
    color = {"attacker": "red", "target": "blue", "judge": "yellow", 
             "finder": "red", "adversary": "magenta", "referee": "green",
             "generator": "cyan"}.get(role.lower(), "white")
    console.print(f"[{color}]{prefix}{role.upper()}:[/{color}] {content[:200]}{'...' if len(content) > 200 else ''}")
