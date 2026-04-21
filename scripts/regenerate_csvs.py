"""Regenerate the three aggregation CSVs from backfilled result JSONs.

Produces:
  - results/thinking_level_sweep.csv
  - results/mixed_model_comparison.csv
  - results/sonnet_self_review_threshold.csv

Does NOT call any LLM. Trial definitions are imported from
runners.full_experiment_batch (the runner that produced the existing
CSVs on disk). Matching is by (finder, adversary, referee, task_spec)
tuple from the stored JSON against the trial's derived tuple.

Row-to-JSON matching rules:
  - Exactly one JSON matches a trial  -> use it
  - Multiple JSONs match one trial    -> use most recent by mtime + warn
  - Multiple trials share a key       -> assign chronologically (oldest
                                         JSON to earliest trial) + warn.
                                         This is how Exp 1 and Exp 3 work:
                                         same 4-tuple across thinking
                                         levels / finder temperatures,
                                         disambiguated only by trial order.
  - No JSON matches a trial           -> write zero-metrics row flagged
                                         with missing_json=true

Usage:
  uv run python scripts/regenerate_csvs.py --dry-run
  uv run python scripts/regenerate_csvs.py
  uv run python scripts/regenerate_csvs.py --csv mixed_model
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from runners.full_experiment_batch import (  # noqa: E402
    MIXED_MODEL_CONFIGS,
    MIXED_MODEL_TASKS,
    MODEL_SONNET,
    SONNET_THRESHOLD_TEMPS,
    TASK_SQL,
    THINKING_SWEEP_TRIALS,
    _a,
    _extract_metrics,
    _g,
)

RESULTS_DIR = _PROJECT_ROOT / "results"


# ── CSV schemas ───────────────────────────────────────────────────────────
#
# Column orders intentionally match the existing CSVs on disk plus the two
# new trailing columns (static_findings + missing_json). For Exp 3, the
# existing CSV had no programmatic_findings column at all, so we add only
# static_findings + missing_json to match the stated backfill goal without
# expanding the schema further than necessary.

EXP_CONFIGS: dict[str, dict[str, Any]] = {
    "thinking_sweep": {
        "csv_name": "thinking_level_sweep.csv",
        "fieldnames": [
            "trial", "model", "thinking_level",
            "issues_found", "issues_confirmed", "adversary_kills", "kill_rate",
            "programmatic_findings", "static_findings", "code_quality_verdict",
            "missing_json",
        ],
    },
    "mixed_model": {
        "csv_name": "mixed_model_comparison.csv",
        "fieldnames": [
            "config_name", "task", "generator", "finder", "adversary", "referee",
            "issues_found", "issues_confirmed", "adversary_kills", "kill_rate",
            "programmatic_findings", "static_findings", "code_quality_verdict",
            "missing_json",
        ],
    },
    "sonnet_threshold": {
        "csv_name": "sonnet_self_review_threshold.csv",
        "fieldnames": [
            "finder_temp", "issues_found", "issues_confirmed",
            "kill_rate", "static_findings", "code_quality_verdict",
            "missing_json",
        ],
    },
}


# ── Trial builders ────────────────────────────────────────────────────────
#
# Each trial dict carries:
#   - metadata: CSV-schema-specific key/value columns
#   - finder/adversary/referee: display-name strings for JSON matching
#   - task_spec: task string for JSON matching

def _build_thinking_sweep_trials() -> list[dict]:
    trials = []
    for trial_num, model, thinking in THINKING_SWEEP_TRIALS:
        finder = _g(model, 1.0, thinking)
        adv    = _g(model, 1.0, thinking)
        ref    = _g(model, 0.1, thinking)
        trials.append({
            "metadata": {
                "trial": trial_num,
                "model": model,
                "thinking_level": thinking,
            },
            "finder":    finder.display_name,
            "adversary": adv.display_name,
            "referee":   ref.display_name,
            "task_spec": TASK_SQL,
        })
    return trials


def _build_mixed_model_trials() -> list[dict]:
    trials = []
    for config_name, gen, finder, adv, ref in MIXED_MODEL_CONFIGS:
        for task_label, task in MIXED_MODEL_TASKS:
            trials.append({
                "metadata": {
                    "config_name": config_name,
                    "task":        task_label,
                    "generator":   gen.display_name,
                    "finder":      finder.display_name,
                    "adversary":   adv.display_name,
                    "referee":     ref.display_name,
                },
                "finder":    finder.display_name,
                "adversary": adv.display_name,
                "referee":   ref.display_name,
                "task_spec": task,
            })
    return trials


def _build_sonnet_threshold_trials() -> list[dict]:
    trials = []
    for finder_temp in SONNET_THRESHOLD_TEMPS:
        finder = _a(MODEL_SONNET, finder_temp)
        adv    = _a(MODEL_SONNET, 0.7)
        ref    = _a(MODEL_SONNET, 0.1)
        trials.append({
            "metadata":  {"finder_temp": finder_temp},
            "finder":    finder.display_name,
            "adversary": adv.display_name,
            "referee":   ref.display_name,
            "task_spec": TASK_SQL,
        })
    return trials


TRIAL_BUILDERS = {
    "thinking_sweep":   _build_thinking_sweep_trials,
    "mixed_model":      _build_mixed_model_trials,
    "sonnet_threshold": _build_sonnet_threshold_trials,
}


# ── JSON indexing ─────────────────────────────────────────────────────────

def _trial_key(entry: dict) -> tuple:
    return (entry["finder"], entry["adversary"], entry["referee"], entry["task_spec"])


def _index_jsons(results_dir: Path) -> tuple[dict[tuple, list[tuple]], list[Path]]:
    """Return (jsons_by_key, malformed_paths).

    jsons_by_key[(finder, adv, ref, task_spec)] = [(mtime, path, data), ...]
    sorted by mtime ascending.
    """
    jsons_by_key: dict[tuple, list[tuple]] = defaultdict(list)
    malformed: list[Path] = []

    for path in sorted(results_dir.rglob("*.json")):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            malformed.append(path)
            continue
        if not isinstance(data, dict):
            malformed.append(path)
            continue
        # code_security JSONs carry all four fields; anything else is ignored
        try:
            key = (
                data["finder"], data["adversary"],
                data["referee"], data["task_spec"],
            )
        except KeyError:
            continue
        mtime = path.stat().st_mtime
        jsons_by_key[key].append((mtime, path, data))

    for key in jsons_by_key:
        jsons_by_key[key].sort(key=lambda x: x[0])

    return jsons_by_key, malformed


# ── Matching ──────────────────────────────────────────────────────────────

def _match_trials(
    trials: list[dict],
    jsons_by_key: dict[tuple, list[tuple]],
) -> list[str]:
    """Attach a `match` field to each trial. Returns list of warning strings."""
    warnings: list[str] = []

    trials_by_key: dict[tuple, list[int]] = defaultdict(list)
    for i, t in enumerate(trials):
        trials_by_key[_trial_key(t)].append(i)

    for key, trial_indices in trials_by_key.items():
        candidates = jsons_by_key.get(key, [])
        key_label = f"finder={key[0]} / task={key[3][:40]!r}"

        if len(trial_indices) == 1:
            i = trial_indices[0]
            if not candidates:
                trials[i]["match"] = None
            elif len(candidates) == 1:
                trials[i]["match"] = candidates[0]
            else:
                warnings.append(
                    f"{len(candidates)} JSONs match single trial "
                    f"({key_label}); using most recent by mtime"
                )
                trials[i]["match"] = candidates[-1]
        else:
            if len(candidates) >= len(trial_indices):
                warnings.append(
                    f"{len(trial_indices)} trials share key ({key_label}); "
                    f"disambiguating by timestamp order"
                )
            else:
                warnings.append(
                    f"{len(trial_indices)} trials share key but only "
                    f"{len(candidates)} JSONs available ({key_label}); "
                    f"{len(trial_indices) - len(candidates)} marked missing"
                )
            for pos, trial_i in enumerate(trial_indices):
                trials[trial_i]["match"] = (
                    candidates[pos] if pos < len(candidates) else None
                )

    return warnings


# ── Row construction ──────────────────────────────────────────────────────

_ZERO_METRICS = {
    "issues_found":         0,
    "issues_confirmed":     0,
    "adversary_kills":      0,
    "kill_rate":            0.0,
    "programmatic_findings": 0,
    "static_findings":       0,
    "code_quality_verdict":  "missing",
}


def _build_rows(trials: list[dict], fieldnames: list[str]) -> tuple[list[dict], int, int]:
    """Return (rows, matched_count, missing_count)."""
    rows: list[dict] = []
    matched = missing = 0

    for t in trials:
        row = dict(t["metadata"])
        if t.get("match") is None:
            row.update(_ZERO_METRICS)
            row["missing_json"] = "true"
            missing += 1
        else:
            _mtime, _path, data = t["match"]
            metrics = _extract_metrics(data)
            row.update(metrics)
            row["missing_json"] = ""
            matched += 1
        rows.append({k: row.get(k, "") for k in fieldnames})

    return rows, matched, missing


# ── Main ──────────────────────────────────────────────────────────────────

def _regenerate_one(
    exp_key: str,
    jsons_by_key: dict[tuple, list[tuple]],
    *,
    results_dir: Path,
    dry_run: bool,
) -> dict[str, Any]:
    cfg = EXP_CONFIGS[exp_key]
    csv_path = results_dir / cfg["csv_name"]
    fieldnames = cfg["fieldnames"]
    trials = TRIAL_BUILDERS[exp_key]()

    warnings = _match_trials(trials, jsons_by_key)
    rows, matched, missing = _build_rows(trials, fieldnames)

    print(f"=== {cfg['csv_name']} ===")
    print(f"  Trials: {len(trials)}  Matched: {matched}  Missing: {missing}")
    if warnings:
        print(f"  Ambiguity warnings ({len(warnings)}):")
        for w in warnings:
            print(f"    - {w}")

    if dry_run:
        print(f"  DRY RUN — would write {len(rows)} rows to {csv_path}")
        # Preview first 3 rows
        preview = rows[:3]
        print(f"  First {len(preview)} row(s) preview:")
        for r in preview:
            print(f"    {r}")
    else:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"  Wrote {len(rows)} rows -> {csv_path}")

    print()
    return {
        "exp_key": exp_key,
        "rows_total": len(rows),
        "matched": matched,
        "missing": missing,
        "warnings": len(warnings),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate aggregation CSVs from backfilled result JSONs."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Scan and report what would be written without touching files",
    )
    parser.add_argument(
        "--csv", choices=sorted(EXP_CONFIGS.keys()), default=None,
        help="Regenerate only one specific CSV (default: all three)",
    )
    parser.add_argument(
        "--results-dir", type=Path, default=RESULTS_DIR,
        help="Directory containing result JSONs (default: ./results)",
    )
    args = parser.parse_args()

    results_dir: Path = args.results_dir
    if not results_dir.is_dir():
        print(f"ERROR: results directory does not exist: {results_dir}", file=sys.stderr)
        return 2

    jsons_by_key, malformed = _index_jsons(results_dir)
    total_jsons = sum(len(v) for v in jsons_by_key.values())
    print(
        f"Indexed {total_jsons} code_security JSON(s) "
        f"across {len(jsons_by_key)} unique (finder, adversary, referee, task) keys"
    )
    if malformed:
        print(f"Skipping {len(malformed)} malformed JSON(s):")
        for p in malformed:
            print(f"  {p.name}")
    if args.dry_run:
        print("DRY RUN — no files will be modified")
    print()

    exp_keys = [args.csv] if args.csv else list(EXP_CONFIGS.keys())

    summaries = []
    for exp_key in exp_keys:
        summaries.append(_regenerate_one(
            exp_key, jsons_by_key,
            results_dir=results_dir, dry_run=args.dry_run,
        ))

    print("=" * 60)
    verb = "would write" if args.dry_run else "wrote"
    for s in summaries:
        print(
            f"{s['exp_key']:<20} {verb} {s['rows_total']:>2} rows  "
            f"(matched={s['matched']}, missing={s['missing']}, "
            f"warnings={s['warnings']})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
