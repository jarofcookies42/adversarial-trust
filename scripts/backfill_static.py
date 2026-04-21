"""Backfill the static_findings field into existing result JSONs.

Runs the bandit-based static scanner over each stored generated_code
string and attaches static_findings + static_severity_score to the JSON.
Does NOT call any LLM, does NOT touch generated_code or any LLM-produced
field, does NOT regenerate results.

The script is idempotent: files that already have a non-empty
static_findings list are skipped unless --force is passed.

Writes are atomic — we serialize to a sibling temp file and rename over
the target so a kill mid-flight can't corrupt a JSON.

Usage:
    uv run python scripts/backfill_static.py --dry-run         # preview
    uv run python scripts/backfill_static.py                   # apply
    uv run python scripts/backfill_static.py --verbose         # per-file findings
    uv run python scripts/backfill_static.py --force           # re-run over already-populated
    uv run python scripts/backfill_static.py --results-dir X   # alt directory
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

# Ensure the project root is on sys.path so we can import domains.* when
# this script is invoked directly (e.g., `python scripts/backfill_static.py`).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from domains.code_security.static_scanner import (  # noqa: E402
    run_static_scanner,
    severity_score,
)


def _iter_result_jsons(results_dir: Path) -> list[Path]:
    """Return every .json under results/ (recursive), sorted by path."""
    return sorted(results_dir.rglob("*.json"))


def _atomic_write_json(path: Path, data: dict) -> None:
    """Write `data` as JSON to `path` atomically via tempfile + rename.

    The temp file lives in the same directory so the rename is atomic on
    the same filesystem. fsync before rename so a power failure between
    write and rename doesn't leave a zero-byte partial.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.stem}.",
        suffix=".json.tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except Exception:
        # Clean up temp on failure; rename already happened otherwise.
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def process_file(
    path: Path,
    *,
    dry_run: bool,
    verbose: bool,
    force: bool,
) -> str:
    """Process one JSON. Returns a status code for tallying:
       'backfilled', 'skipped_no_code', 'skipped_has_static',
       'skipped_malformed', 'no_findings_but_noted'
    """
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  [MALFORMED] {path.name}: {e}")
        return "skipped_malformed"

    if not isinstance(data, dict):
        print(f"  [SKIP] {path.name}: top-level is not an object")
        return "skipped_malformed"

    code = data.get("generated_code")
    if not isinstance(code, str) or not code.strip():
        if verbose:
            print(f"  [SKIP-no-code] {path.name}")
        return "skipped_no_code"

    existing = data.get("static_findings")
    if existing and not force:
        if verbose:
            print(
                f"  [SKIP-has-static] {path.name} "
                f"(already has {len(existing)} static findings)"
            )
        return "skipped_has_static"

    findings = run_static_scanner(code)
    score = severity_score(findings)

    if verbose:
        print(f"  [SCAN] {path.name}: {len(findings)} findings, severity_score={score}")
        for f in findings:
            print(
                f"         [{f['severity']:<8}] {f['rule_id']:<5} "
                f"{f['category']:<10} L{f['line']:<4} {f['description'][:70]}"
            )
    else:
        print(f"  [SCAN] {path.name}: {len(findings):>2} findings (score={score:>3})")

    if dry_run:
        return "backfilled"  # count as would-backfill

    data["static_findings"] = findings
    data["static_severity_score"] = score
    _atomic_write_json(path, data)

    return "backfilled"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill static_findings into existing result JSONs by running "
            "the static scanner over each stored generated_code."
        )
    )
    parser.add_argument(
        "--results-dir", type=Path,
        default=_PROJECT_ROOT / "results",
        help="Directory containing result JSONs (default: ./results)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Scan but don't write any file",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print each finding per file",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-scan files that already have static_findings (default: skip them)",
    )
    args = parser.parse_args()

    results_dir: Path = args.results_dir
    if not results_dir.is_dir():
        print(f"ERROR: results directory does not exist: {results_dir}", file=sys.stderr)
        return 2

    jsons = _iter_result_jsons(results_dir)
    print(f"Found {len(jsons)} JSON file(s) under {results_dir}")
    if args.dry_run:
        print("DRY RUN — no files will be modified")
    if args.force:
        print("FORCE — rescanning files that already have static_findings")
    print()

    tally = {
        "backfilled": 0,
        "skipped_no_code": 0,
        "skipped_has_static": 0,
        "skipped_malformed": 0,
    }

    for path in jsons:
        status = process_file(
            path,
            dry_run=args.dry_run,
            verbose=args.verbose,
            force=args.force,
        )
        tally[status] = tally.get(status, 0) + 1

    print()
    print("=" * 60)
    verb = "would backfill" if args.dry_run else "backfilled"
    print(
        f"Processed {len(jsons)} files: "
        f"{tally['backfilled']} {verb}, "
        f"{tally['skipped_no_code']} skipped (no code), "
        f"{tally['skipped_has_static']} skipped (already had static_findings), "
        f"{tally['skipped_malformed']} skipped (malformed)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
