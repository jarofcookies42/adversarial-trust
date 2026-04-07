"""Batch runner for code security adversarial reviews.

Runs multiple trials with different task specs and model configurations,
saves a summary CSV, and prints a results table.

Usage:
    python -m runners.batch_code_security              # original 7-trial batch
    python -m runners.batch_code_security --tiers       # 12-trial tier comparison
"""

from __future__ import annotations

import argparse
import csv
import json
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table

from maat.config import MODELS, RECURSION_LIMIT, RESULTS_DIR

console = Console()


@dataclass
class TrialConfig:
    """Configuration for a single trial."""
    task: str
    generator: str
    finder: str | None = None
    adversary: str | None = None
    referee: str | None = None
    tier: str = ""

    @property
    def finder_display(self) -> str:
        return self.finder or self.generator

    @property
    def adversary_display(self) -> str:
        return self.adversary or self.generator

    @property
    def referee_display(self) -> str:
        return self.referee or self.generator


@dataclass
class TrialResult:
    """Result summary for a single trial."""
    task: str
    generator: str
    finder: str
    adversary: str
    referee: str
    issues_found: int
    issues_confirmed: int
    issues_rejected: int
    programmatic_findings: int
    code_quality_verdict: str
    tier: str = ""
    error: str = ""


def run_trial(config: TrialConfig) -> TrialResult:
    """Run a single code security trial and return its summary."""
    from domains.code_security.graph import build_code_security_graph

    generator_config = MODELS[config.generator]
    generator_config.temperature = 0.3

    finder_config = MODELS.get(config.finder) if config.finder else None
    adversary_config = MODELS.get(config.adversary) if config.adversary else None
    referee_config = MODELS.get(config.referee) if config.referee else None

    graph = build_code_security_graph(
        generator_config=generator_config,
        finder_config=finder_config,
        adversary_config=adversary_config,
        referee_config=referee_config,
    )

    initial_state = {
        "domain": "code_security",
        "task_spec": config.task,
        "conversation_history": [],
        "judge_evaluations": [],
        "round_number": 0,
        "max_rounds": 1,
        "metadata": {"generator": generator_config.display_name},
    }

    result = graph.invoke(initial_state, {"recursion_limit": RECURSION_LIMIT})

    # Extract from result JSON (mirrors what graph.py saves)
    verdict = result.get("referee_verdict", {})
    summary = verdict.get("summary", {}) if isinstance(verdict, dict) else {}
    fi = verdict.get("final_issues", []) if isinstance(verdict, dict) else []
    prog = result.get("programmatic_findings", [])
    finder_issues = result.get("finder_issues", [])

    # Use summary if available, otherwise count from final_issues
    found = summary.get("total_found") or len(finder_issues)
    confirmed = summary.get("confirmed") or sum(
        1 for i in fi if isinstance(i, dict) and i.get("final_verdict") == "confirmed")
    rejected = summary.get("rejected") or sum(
        1 for i in fi if isinstance(i, dict) and i.get("final_verdict") == "rejected")

    return TrialResult(
        task=config.task,
        generator=config.generator,
        finder=config.finder_display,
        adversary=config.adversary_display,
        referee=config.referee_display,
        issues_found=found,
        issues_confirmed=confirmed,
        issues_rejected=rejected,
        programmatic_findings=len(prog),
        code_quality_verdict=verdict.get("overall_code_quality", "unknown")
            if isinstance(verdict, dict) else "unknown",
        tier=config.tier,
    )


def run_batch(trials: list[TrialConfig]) -> list[TrialResult]:
    """Execute a list of trials, handling errors gracefully."""
    results: list[TrialResult] = []
    total = len(trials)

    for i, trial in enumerate(trials, 1):
        tag = f"[{i}/{total}]"
        tier_tag = f" [{trial.tier}]" if trial.tier else ""
        console.print(f"\n{'='*70}")
        console.print(f"[bold cyan]{tag}{tier_tag} Task:[/bold cyan] {trial.task[:80]}")
        console.print(f"[dim]Model={trial.generator} (all roles)[/dim]")
        console.print(f"{'='*70}")

        try:
            r = run_trial(trial)
            results.append(r)
            console.print(f"[green]{tag} Done:[/green] {r.issues_found} found, "
                          f"{r.issues_confirmed} confirmed, {r.issues_rejected} rejected, "
                          f"{r.programmatic_findings} programmatic, quality={r.code_quality_verdict}")
        except Exception as e:
            console.print(f"[red]{tag} FAILED:[/red] {e}")
            traceback.print_exc()
            results.append(TrialResult(
                task=trial.task, generator=trial.generator,
                finder=trial.finder_display, adversary=trial.adversary_display,
                referee=trial.referee_display, tier=trial.tier,
                issues_found=0, issues_confirmed=0, issues_rejected=0,
                programmatic_findings=0, code_quality_verdict="error",
                error=str(e),
            ))

    return results


def save_csv(results: list[TrialResult], csv_path: Path) -> None:
    """Save trial results to CSV."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "tier", "task", "generator", "finder", "adversary", "referee",
        "issues_found", "issues_confirmed", "issues_rejected",
        "programmatic_findings", "code_quality_verdict", "error",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "tier": r.tier, "task": r.task,
                "generator": r.generator, "finder": r.finder,
                "adversary": r.adversary, "referee": r.referee,
                "issues_found": r.issues_found,
                "issues_confirmed": r.issues_confirmed,
                "issues_rejected": r.issues_rejected,
                "programmatic_findings": r.programmatic_findings,
                "code_quality_verdict": r.code_quality_verdict,
                "error": r.error,
            })
    console.print(f"\n[green]CSV saved:[/green] {csv_path}")


def print_tier_table(results: list[TrialResult]) -> None:
    """Print a summary table grouped by tier."""
    table = Table(title="MAAT Tier Comparison — Code Security", show_lines=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Tier", width=13)
    table.add_column("Model", width=22)
    table.add_column("Task", max_width=28)
    table.add_column("Found", justify="right", width=6)
    table.add_column("Conf.", justify="right", width=6)
    table.add_column("Rej.", justify="right", width=5)
    table.add_column("Kill%", justify="right", width=6)
    table.add_column("Prog.", justify="right", width=6)
    table.add_column("Quality", width=12)

    tier_order = {"ultra-fast": 0, "goldilocks": 1, "heavyweight": 2}
    sorted_results = sorted(results, key=lambda r: (tier_order.get(r.tier, 9), r.task, r.generator))

    for i, r in enumerate(sorted_results, 1):
        qc = r.code_quality_verdict
        color = {"secure": "green", "mostly_secure": "green", "concerning": "yellow",
                 "vulnerable": "red", "critical": "bold red", "error": "dim red"}.get(qc, "white")
        tier_color = {"ultra-fast": "cyan", "goldilocks": "yellow", "heavyweight": "magenta"}.get(r.tier, "white")
        kill_rate = f"{r.issues_rejected / r.issues_found * 100:.0f}%" if r.issues_found > 0 else "-"
        model_short = r.generator

        task_short = "SQL queries" if "PostgreSQL" in r.task else "URL fetcher"

        table.add_row(
            str(i),
            f"[{tier_color}]{r.tier}[/{tier_color}]",
            model_short,
            task_short,
            str(r.issues_found),
            str(r.issues_confirmed),
            str(r.issues_rejected),
            kill_rate,
            str(r.programmatic_findings),
            f"[{color}]{qc}[/{color}]",
        )

    console.print()
    console.print(table)

    # Per-tier averages
    console.print("\n[bold underline]Per-tier averages:[/bold underline]")
    for tier in ["ultra-fast", "goldilocks", "heavyweight"]:
        tier_results = [r for r in results if r.tier == tier and r.error == ""]
        if not tier_results:
            continue
        # Group by model
        models = sorted(set(r.generator for r in tier_results))
        tier_color = {"ultra-fast": "cyan", "goldilocks": "yellow", "heavyweight": "magenta"}[tier]
        console.print(f"\n  [{tier_color}]{tier.upper()}:[/{tier_color}]")
        for model in models:
            mr = [r for r in tier_results if r.generator == model]
            n = len(mr)
            avg_found = sum(r.issues_found for r in mr) / n
            avg_conf = sum(r.issues_confirmed for r in mr) / n
            avg_rej = sum(r.issues_rejected for r in mr) / n
            avg_prog = sum(r.programmatic_findings for r in mr) / n
            total_found = sum(r.issues_found for r in mr)
            total_rej = sum(r.issues_rejected for r in mr)
            kill_pct = f"{total_rej / total_found * 100:.0f}%" if total_found > 0 else "-"
            console.print(f"    {model:26s}  found={avg_found:5.1f}  confirmed={avg_conf:5.1f}  "
                          f"kill_rate={kill_pct:>4s}  programmatic={avg_prog:4.1f}")


def build_tier_trials() -> list[TrialConfig]:
    """Build the 12-trial tier comparison config."""
    tasks = [
        "Write a Python script that connects to a PostgreSQL database and runs user-provided SQL queries",
        "Build an API that fetches data from external URLs provided by the user and returns the response",
    ]

    trials: list[TrialConfig] = []

    # Tier 1 — Ultra-fast
    for model in ["haiku", "gemini-3.1-flash-lite"]:
        for task in tasks:
            trials.append(TrialConfig(task=task, generator=model, tier="ultra-fast"))

    # Tier 2 — Goldilocks
    for model in ["sonnet", "gemini-3-flash"]:
        for task in tasks:
            trials.append(TrialConfig(task=task, generator=model, tier="goldilocks"))

    # Tier 3 — Heavyweight
    for model in ["opus", "gemini-3.1-pro"]:
        for task in tasks:
            trials.append(TrialConfig(task=task, generator=model, tier="heavyweight"))

    return trials


def build_original_trials() -> list[TrialConfig]:
    """Build the original 7-trial batch config."""
    tasks = [
        "Implement a user registration and login system with password hashing in Python Flask",
        "Build a REST API endpoint that accepts file uploads and stores them on disk",
        "Write a Python script that connects to a PostgreSQL database and runs user-provided SQL queries",
        "Implement a session management system with cookie-based authentication",
        "Build an API that fetches data from external URLs provided by the user and returns the response",
    ]

    trials: list[TrialConfig] = []
    for task in tasks:
        trials.append(TrialConfig(task=task, generator="gemini-2.5-flash"))
    for task in [tasks[2], tasks[4]]:
        trials.append(TrialConfig(
            task=task, generator="gemini-2.5-flash",
            finder="sonnet", adversary="sonnet", referee="gemini-2.5-flash",
        ))
    return trials


def main() -> None:
    parser = argparse.ArgumentParser(description="MAAT batch code security runner")
    parser.add_argument("--tiers", action="store_true", help="Run 12-trial tier comparison")
    args = parser.parse_args()

    if args.tiers:
        trials = build_tier_trials()
        results = run_batch(trials)
        csv_path = RESULTS_DIR / "tier_comparison.csv"
        save_csv(results, csv_path)
        print_tier_table(results)
    else:
        trials = build_original_trials()
        results = run_batch(trials)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = RESULTS_DIR / f"batch_summary_{ts}.csv"
        save_csv(results, csv_path)

        # Simple table for original mode
        table = Table(title="Code Security Batch Results", show_lines=True)
        table.add_column("#", style="dim", width=3)
        table.add_column("Task", max_width=45)
        table.add_column("Finder", width=12)
        table.add_column("Found", justify="right", width=6)
        table.add_column("Confirmed", justify="right", width=9)
        table.add_column("Rejected", justify="right", width=8)
        table.add_column("Prog.", justify="right", width=6)
        table.add_column("Quality", width=14)

        for i, r in enumerate(results, 1):
            qc_color = {"secure": "green", "mostly_secure": "green", "concerning": "yellow",
                        "vulnerable": "red", "critical": "bold red", "error": "dim red"}.get(
                        r.code_quality_verdict, "white")
            table.add_row(str(i), r.task[:45], r.finder, str(r.issues_found),
                          str(r.issues_confirmed), str(r.issues_rejected),
                          str(r.programmatic_findings),
                          f"[{qc_color}]{r.code_quality_verdict}[/{qc_color}]")

        console.print()
        console.print(table)


if __name__ == "__main__":
    main()
