"""Re-run Goldilocks and Heavyweight tiers with thinking level support.

Tests Gemini 3.x models at two finder temperatures (0.7 standard, 1.0 Google-recommended)
alongside Anthropic models at 0.7, on the PostgreSQL SQL queries task.

8 trials total:
- Sonnet @ finder 0.7 (baseline)
- Gemini 3 Flash @ finder 0.7 (thinking=high)
- Gemini 3 Flash @ finder 1.0 (thinking=high)
- Sonnet @ finder 0.7 (baseline, repeated for Heavyweight comparison)
  -> actually just Opus, since we're comparing within tiers
- Opus @ finder 0.7 (baseline)
- Gemini 3.1 Pro @ finder 0.7 (thinking=high)
- Gemini 3.1 Pro @ finder 1.0 (thinking=high)
- Opus @ finder 0.7 (baseline, but no need to repeat — just run once)

Actual 8 trials:
  Goldilocks: Sonnet 0.7, Flash 0.7, Flash 1.0, Sonnet 1.0 (for symmetry? no)
  -> Sonnet 0.7, Flash 0.7, Flash 1.0 = 3 trials
  Heavyweight: Opus 0.7, Pro 0.7, Pro 1.0 = 3 trials
  That's 6. To get 8: run each Anthropic model at 0.7 only, each Google at 0.7 AND 1.0.
  Goldilocks: Sonnet 0.7, Flash 0.7, Flash 1.0 = 3
  Heavyweight: Opus 0.7, Pro 0.7, Pro 1.0 = 3
  That's 6. Add Sonnet 1.0 and Opus 1.0 too? No, Anthropic caps at 1.0.
  -> Run Anthropic at 0.7 once each, Google at 0.7 and 1.0 each = 8 total.
"""

from __future__ import annotations

import csv
import traceback
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.table import Table

from maat.config import MODELS, RECURSION_LIMIT, RESULTS_DIR, ModelConfig

console = Console()

TASK = "Write a Python script that connects to a PostgreSQL database and runs user-provided SQL queries"


@dataclass
class Trial:
    tier: str
    model_key: str
    finder_temp: float
    thinking_level: str | None = None


@dataclass
class Result:
    tier: str
    model: str
    finder_temp: float
    thinking_level: str
    issues_found: int
    issues_confirmed: int
    issues_rejected: int
    kill_rate: float
    programmatic_findings: int
    code_quality_verdict: str
    error: str = ""


def make_config(model_key: str, temperature: float, thinking_level: str | None = None) -> ModelConfig:
    """Create a ModelConfig with specific overrides."""
    base = MODELS[model_key]
    return ModelConfig(
        provider=base.provider,
        model_name=base.model_name,
        temperature=temperature,
        max_tokens=base.max_tokens,
        thinking_level=thinking_level if thinking_level is not None else base.thinking_level,
    )


def run_trial(trial: Trial) -> Result:
    from domains.code_security.graph import build_code_security_graph

    gen_cfg = make_config(trial.model_key, 0.3, trial.thinking_level)
    finder_cfg = make_config(trial.model_key, trial.finder_temp, trial.thinking_level)
    adv_cfg = make_config(trial.model_key, 0.7, trial.thinking_level)
    ref_cfg = make_config(trial.model_key, 0.1, trial.thinking_level)

    graph = build_code_security_graph(
        generator_config=gen_cfg,
        finder_config=finder_cfg,
        adversary_config=adv_cfg,
        referee_config=ref_cfg,
    )

    initial_state = {
        "domain": "code_security",
        "task_spec": TASK,
        "conversation_history": [],
        "judge_evaluations": [],
        "round_number": 0,
        "max_rounds": 1,
        "metadata": {
            "generator": gen_cfg.display_name,
            "finder_temp": trial.finder_temp,
            "thinking_level": trial.thinking_level or "N/A",
        },
    }

    result = graph.invoke(initial_state, {"recursion_limit": RECURSION_LIMIT})

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
    quality = verdict.get("overall_code_quality", "unknown") if isinstance(verdict, dict) else "unknown"

    return Result(
        tier=trial.tier,
        model=trial.model_key,
        finder_temp=trial.finder_temp,
        thinking_level=trial.thinking_level or "N/A",
        issues_found=found,
        issues_confirmed=confirmed,
        issues_rejected=rejected,
        kill_rate=round(kill, 1),
        programmatic_findings=len(prog),
        code_quality_verdict=quality,
    )


def main() -> None:
    trials = [
        # Goldilocks tier
        Trial("goldilocks", "sonnet",        0.7),
        Trial("goldilocks", "gemini-3-flash", 0.7, "high"),
        Trial("goldilocks", "gemini-3-flash", 1.0, "high"),
        Trial("goldilocks", "sonnet",        0.7),  # repeat for variance check
        # Heavyweight tier
        Trial("heavyweight", "opus",          0.7),
        Trial("heavyweight", "gemini-3.1-pro", 0.7, "high"),
        Trial("heavyweight", "gemini-3.1-pro", 1.0, "high"),
        Trial("heavyweight", "opus",          0.7),  # repeat for variance check
    ]

    results: list[Result] = []
    total = len(trials)

    for i, t in enumerate(trials, 1):
        tag = f"[{i}/{total}]"
        think_tag = f" thinking={t.thinking_level}" if t.thinking_level else ""
        console.print(f"\n{'='*70}")
        console.print(f"[bold cyan]{tag} {t.tier.upper()} — {t.model_key} "
                      f"finder_temp={t.finder_temp}{think_tag}[/bold cyan]")
        console.print(f"{'='*70}")

        try:
            r = run_trial(t)
            results.append(r)
            console.print(f"[green]{tag} Done:[/green] found={r.issues_found} "
                          f"confirmed={r.issues_confirmed} rejected={r.issues_rejected} "
                          f"kill={r.kill_rate}% prog={r.programmatic_findings} "
                          f"quality={r.code_quality_verdict}")
        except Exception as e:
            console.print(f"[red]{tag} FAILED:[/red] {e}")
            traceback.print_exc()
            results.append(Result(
                tier=t.tier, model=t.model_key, finder_temp=t.finder_temp,
                thinking_level=t.thinking_level or "N/A",
                issues_found=0, issues_confirmed=0, issues_rejected=0,
                kill_rate=0.0, programmatic_findings=0,
                code_quality_verdict="error", error=str(e),
            ))

    # Save CSV
    csv_path = RESULTS_DIR / "thinking_level_rerun.csv"
    fieldnames = ["tier", "model", "finder_temp", "thinking_level",
                  "issues_found", "issues_confirmed", "issues_rejected",
                  "kill_rate", "programmatic_findings", "code_quality_verdict", "error"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow({
                "tier": r.tier, "model": r.model,
                "finder_temp": r.finder_temp, "thinking_level": r.thinking_level,
                "issues_found": r.issues_found, "issues_confirmed": r.issues_confirmed,
                "issues_rejected": r.issues_rejected, "kill_rate": r.kill_rate,
                "programmatic_findings": r.programmatic_findings,
                "code_quality_verdict": r.code_quality_verdict, "error": r.error,
            })
    console.print(f"\n[green]CSV saved:[/green] {csv_path}")

    # Print table grouped by tier
    for tier in ["goldilocks", "heavyweight"]:
        tier_results = [r for r in results if r.tier == tier]
        tier_color = {"goldilocks": "yellow", "heavyweight": "magenta"}[tier]

        table = Table(title=f"{tier.upper()} Tier — Thinking Level Rerun (Task: SQL Queries)",
                      show_lines=True)
        table.add_column("Model", width=18)
        table.add_column("Finder T", width=9)
        table.add_column("Thinking", width=9)
        table.add_column("Found", justify="right", width=6)
        table.add_column("Conf.", justify="right", width=6)
        table.add_column("Rej.", justify="right", width=5)
        table.add_column("Kill%", justify="right", width=6)
        table.add_column("Prog.", justify="right", width=6)
        table.add_column("Quality", width=14)

        for r in tier_results:
            qc = r.code_quality_verdict
            color = {"secure": "green", "mostly_secure": "green", "concerning": "yellow",
                     "vulnerable": "red", "critical": "bold red", "error": "dim red"}.get(qc, "white")
            table.add_row(
                r.model, f"{r.finder_temp}", r.thinking_level,
                str(r.issues_found), str(r.issues_confirmed), str(r.issues_rejected),
                f"{r.kill_rate:.0f}%", str(r.programmatic_findings),
                f"[{color}]{qc}[/{color}]",
            )

        console.print()
        console.print(table)

    # Comparison vs previous tier results
    console.print("\n[bold underline]Comparison with previous tier run (no thinking, default temps):[/bold underline]")
    console.print("[dim]Previous: Gemini 3 Flash found 4.0 avg, Gemini 3.1 Pro found 2.0 avg[/dim]")
    for model in ["gemini-3-flash", "gemini-3.1-pro"]:
        mr = [r for r in results if r.model == model and r.error == ""]
        if mr:
            for r in mr:
                console.print(f"  {model} (temp={r.finder_temp}, think={r.thinking_level}): "
                              f"found={r.issues_found} confirmed={r.issues_confirmed}")


if __name__ == "__main__":
    main()
