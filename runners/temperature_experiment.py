"""Temperature sensitivity experiment for code security domain.

Tests how varying the Finder temperature affects issue discovery,
with fixed temperatures for Generator (0.3), Adversary (0.9), Referee (0.1).

Also tests the Sonnet self-review bias with higher temperatures.
"""

from __future__ import annotations

import csv
import json
import traceback
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.table import Table

from maat.config import MODELS, RECURSION_LIMIT, RESULTS_DIR, ModelConfig, Provider

console = Console()

TASK = "Write a Python script that connects to a PostgreSQL database and runs user-provided SQL queries"


@dataclass
class TempTrialResult:
    """Result for a single temperature trial."""
    model: str
    finder_temp: float
    issues_found: int
    issues_confirmed: int
    issues_rejected: int
    kill_rate: float
    programmatic_findings: int
    code_quality_verdict: str
    error: str = ""


def make_config(model_key: str, temperature: float) -> ModelConfig:
    """Create a ModelConfig with a specific temperature override."""
    base = MODELS[model_key]
    return ModelConfig(
        provider=base.provider,
        model_name=base.model_name,
        temperature=temperature,
        max_tokens=base.max_tokens,
    )


def run_temp_trial(model_key: str, finder_temp: float) -> TempTrialResult:
    """Run a single trial with a specific finder temperature."""
    from domains.code_security.graph import build_code_security_graph

    generator_cfg = make_config(model_key, 0.3)
    finder_cfg = make_config(model_key, finder_temp)
    adversary_cfg = make_config(model_key, 0.9)
    referee_cfg = make_config(model_key, 0.1)

    graph = build_code_security_graph(
        generator_config=generator_cfg,
        finder_config=finder_cfg,
        adversary_config=adversary_cfg,
        referee_config=referee_cfg,
    )

    initial_state = {
        "domain": "code_security",
        "task_spec": TASK,
        "conversation_history": [],
        "judge_evaluations": [],
        "round_number": 0,
        "max_rounds": 1,
        "metadata": {"generator": generator_cfg.display_name, "finder_temp": finder_temp},
    }

    result = graph.invoke(initial_state, {"recursion_limit": RECURSION_LIMIT})

    # Extract metrics
    verdict = result.get("referee_verdict", {})
    summary = verdict.get("summary", {}) if isinstance(verdict, dict) else {}
    fi = verdict.get("final_issues", []) if isinstance(verdict, dict) else []
    finder_issues = result.get("finder_issues", [])
    prog = result.get("programmatic_findings", [])

    found = summary.get("total_found") or len(finder_issues)
    confirmed = summary.get("confirmed") or sum(
        1 for i in fi if isinstance(i, dict) and i.get("final_verdict") == "confirmed")
    rejected = summary.get("rejected") or sum(
        1 for i in fi if isinstance(i, dict) and i.get("final_verdict") == "rejected")
    kill = rejected / found * 100 if found > 0 else 0.0

    quality = (verdict.get("overall_code_quality", "unknown")
               if isinstance(verdict, dict) else "unknown")

    return TempTrialResult(
        model=model_key,
        finder_temp=finder_temp,
        issues_found=found,
        issues_confirmed=confirmed,
        issues_rejected=rejected,
        kill_rate=round(kill, 1),
        programmatic_findings=len(prog),
        code_quality_verdict=quality,
    )


def main() -> None:
    # Part 1: Opus temperature sweep
    opus_temps = [0.1, 0.3, 0.5, 0.7, 0.9, 1.2]
    # Part 2: Sonnet self-review bias test
    sonnet_temps = [0.9, 1.2]

    trials = [(m, t) for m, t in
              [("opus", t) for t in opus_temps] +
              [("sonnet", t) for t in sonnet_temps]]

    results: list[TempTrialResult] = []
    total = len(trials)

    for i, (model, temp) in enumerate(trials, 1):
        tag = f"[{i}/{total}]"
        console.print(f"\n{'='*70}")
        console.print(f"[bold cyan]{tag} {model.upper()} — finder_temp={temp}[/bold cyan]")
        console.print(f"[dim]gen=0.3 adv=0.9 ref=0.1 | Task: SQL queries[/dim]")
        console.print(f"{'='*70}")

        try:
            r = run_temp_trial(model, temp)
            results.append(r)
            console.print(f"[green]{tag} Done:[/green] found={r.issues_found} "
                          f"confirmed={r.issues_confirmed} rejected={r.issues_rejected} "
                          f"kill={r.kill_rate}% prog={r.programmatic_findings} "
                          f"quality={r.code_quality_verdict}")
        except Exception as e:
            console.print(f"[red]{tag} FAILED:[/red] {e}")
            traceback.print_exc()
            results.append(TempTrialResult(
                model=model, finder_temp=temp,
                issues_found=0, issues_confirmed=0, issues_rejected=0,
                kill_rate=0.0, programmatic_findings=0,
                code_quality_verdict="error", error=str(e),
            ))

    # Save CSV
    csv_path = RESULTS_DIR / "temperature_experiment.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model", "finder_temp", "issues_found", "issues_confirmed",
                  "issues_rejected", "kill_rate", "programmatic_findings",
                  "code_quality_verdict", "error"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "model": r.model, "finder_temp": r.finder_temp,
                "issues_found": r.issues_found, "issues_confirmed": r.issues_confirmed,
                "issues_rejected": r.issues_rejected, "kill_rate": r.kill_rate,
                "programmatic_findings": r.programmatic_findings,
                "code_quality_verdict": r.code_quality_verdict, "error": r.error,
            })
    console.print(f"\n[green]CSV saved:[/green] {csv_path}")

    # Part 1 table: Opus temperature sweep
    opus_results = [r for r in results if r.model == "opus"]
    table = Table(title="Opus 4.6 — Finder Temperature Sensitivity (Task: SQL Queries)",
                  show_lines=True)
    table.add_column("Finder Temp", width=12)
    table.add_column("Found", justify="right", width=6)
    table.add_column("Confirmed", justify="right", width=10)
    table.add_column("Rejected", justify="right", width=9)
    table.add_column("Kill Rate", justify="right", width=10)
    table.add_column("Prog.", justify="right", width=6)
    table.add_column("Quality", width=14)

    for r in opus_results:
        qc = r.code_quality_verdict
        color = {"secure": "green", "mostly_secure": "green", "concerning": "yellow",
                 "vulnerable": "red", "critical": "bold red", "error": "dim red"}.get(qc, "white")
        temp_label = f"{r.finder_temp:.1f}"
        if r.finder_temp == 0.9:
            temp_label += " (default)"
        table.add_row(
            temp_label, str(r.issues_found), str(r.issues_confirmed),
            str(r.issues_rejected), f"{r.kill_rate:.0f}%",
            str(r.programmatic_findings), f"[{color}]{qc}[/{color}]",
        )

    console.print()
    console.print(table)

    # Part 2 table: Sonnet self-review bias
    sonnet_results = [r for r in results if r.model == "sonnet"]
    table2 = Table(title="Sonnet 4.6 — Self-Review Bias Test (Task: SQL Queries)",
                   show_lines=True)
    table2.add_column("Finder Temp", width=12)
    table2.add_column("Found", justify="right", width=6)
    table2.add_column("Confirmed", justify="right", width=10)
    table2.add_column("Rejected", justify="right", width=9)
    table2.add_column("Kill Rate", justify="right", width=10)
    table2.add_column("Prog.", justify="right", width=6)
    table2.add_column("Quality", width=14)

    for r in sonnet_results:
        qc = r.code_quality_verdict
        color = {"secure": "green", "mostly_secure": "green", "concerning": "yellow",
                 "vulnerable": "red", "critical": "bold red", "error": "dim red"}.get(qc, "white")
        table2.add_row(
            f"{r.finder_temp:.1f}", str(r.issues_found), str(r.issues_confirmed),
            str(r.issues_rejected), f"{r.kill_rate:.0f}%",
            str(r.programmatic_findings), f"[{color}]{qc}[/{color}]",
        )

    console.print()
    console.print(table2)

    # Analysis
    console.print("\n[bold underline]Analysis:[/bold underline]")
    if opus_results:
        found_by_temp = [(r.finder_temp, r.issues_found) for r in opus_results if r.error == ""]
        if found_by_temp:
            min_r = min(found_by_temp, key=lambda x: x[1])
            max_r = max(found_by_temp, key=lambda x: x[1])
            console.print(f"  Opus: fewest issues at temp {min_r[0]} ({min_r[1]}), "
                          f"most at temp {max_r[0]} ({max_r[1]})")

    if sonnet_results:
        prev_zero = "Sonnet found 0 issues at temp 0.9 in tier comparison (self-review)"
        for r in sonnet_results:
            if r.error == "":
                delta = "still 0" if r.issues_found == 0 else f"now {r.issues_found}"
                console.print(f"  Sonnet temp {r.finder_temp}: {delta} issues "
                              f"(bias {'persists' if r.issues_found == 0 else 'BROKEN'})")


if __name__ == "__main__":
    main()
