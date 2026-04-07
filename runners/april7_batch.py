"""April 7, 2026 — MAAT Experiment Batch.

Three experiments, 27 trials total:
  Exp 1 — Gemini Thinking-Level Sweep (9 trials)
  Exp 2 — Cross-Provider Mixed-Model (12 trials: 6 configs × 2 tasks)
  Exp 3 — Sonnet Self-Review Temperature Threshold (6 trials)

Run with:
  python -m runners.april7_batch
"""

from __future__ import annotations

import csv
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.table import Table

from maat.config import MODELS, RECURSION_LIMIT, RESULTS_DIR, Provider, ModelConfig

console = Console()

# ── Tasks ──────────────────────────────────────────────────────────────────────
TASK_A = (
    "Write a Python script that connects to a PostgreSQL database and runs "
    "user-provided SQL queries"
)
TASK_B = (
    "Build an API that fetches data from external URLs provided by the user "
    "and returns the response"
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def mc(provider: Provider, model_name: str, temperature: float,
       thinking_level: str | None = None, max_tokens: int = 4096) -> ModelConfig:
    """Shorthand ModelConfig constructor."""
    return ModelConfig(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        thinking_level=thinking_level,
    )


# Canonical model names
GOOGLE = Provider.GOOGLE
ANTHROPIC = Provider.ANTHROPIC

FLASH      = "gemini-3-flash-preview"
PRO        = "gemini-3.1-pro-preview"
FLASH_LITE = "gemini-3.1-flash-lite-preview"
SONNET     = "claude-sonnet-4-6"
OPUS       = "claude-opus-4-6"
HAIKU      = "claude-haiku-4-5-20251001"


def _extract_metrics(graph_result: dict) -> dict:
    """Pull standard metrics from a graph result dict."""
    verdict = graph_result.get("referee_verdict", {}) or {}
    if not isinstance(verdict, dict):
        verdict = {}

    summary = verdict.get("summary", {}) or {}
    final_issues = verdict.get("final_issues", []) or []
    finder_issues = graph_result.get("finder_issues", []) or []
    adversary_challenges = graph_result.get("adversary_challenges", []) or []
    prog = graph_result.get("programmatic_findings", []) or []

    found = summary.get("total_found") or len(finder_issues)
    confirmed = summary.get("confirmed") or sum(
        1 for i in final_issues
        if isinstance(i, dict) and i.get("final_verdict") == "confirmed"
    )
    rejected = summary.get("rejected") or sum(
        1 for i in final_issues
        if isinstance(i, dict) and i.get("final_verdict") == "rejected"
    )
    adversary_kills = sum(
        1 for c in adversary_challenges
        if isinstance(c, dict) and c.get("verdict") == "killed"
    )
    kill_rate = round(adversary_kills / found * 100, 1) if found > 0 else 0.0
    quality = verdict.get("overall_code_quality", "unknown")

    return {
        "issues_found": found,
        "issues_confirmed": confirmed,
        "adversary_kills": adversary_kills,
        "kill_rate": kill_rate,
        "programmatic_findings": len(prog),
        "code_quality_verdict": quality,
    }


def run_trial(generator_config: ModelConfig, finder_config: ModelConfig,
              adversary_config: ModelConfig, referee_config: ModelConfig,
              task_spec: str) -> dict:
    """Build graph, invoke, return metrics dict. Raises on failure."""
    from domains.code_security.graph import build_code_security_graph

    graph = build_code_security_graph(
        generator_config=generator_config,
        finder_config=finder_config,
        adversary_config=adversary_config,
        referee_config=referee_config,
    )

    initial_state = {
        "domain": "code_security",
        "task_spec": task_spec,
        "conversation_history": [],
        "judge_evaluations": [],
        "round_number": 0,
        "max_rounds": 1,
        "metadata": {
            "generator": generator_config.display_name,
            "finder": finder_config.display_name,
            "adversary": adversary_config.display_name,
            "referee": referee_config.display_name,
        },
    }

    result = graph.invoke(initial_state, {"recursion_limit": RECURSION_LIMIT})
    return _extract_metrics(result)


def _err_row(**kwargs) -> dict:
    """Build a zero-metrics row for a failed trial."""
    return {
        "issues_found": 0,
        "issues_confirmed": 0,
        "adversary_kills": 0,
        "kill_rate": 0.0,
        "programmatic_findings": 0,
        "code_quality_verdict": "error",
        **kwargs,
    }


def _print_banner(trial_num: int, total: int, label: str) -> None:
    console.print(f"\n{'='*72}")
    console.print(f"[bold cyan]Trial {trial_num}/{total}: {label}[/bold cyan]")
    console.print(f"{'='*72}")


# ── Experiment 1 ───────────────────────────────────────────────────────────────

def run_experiment_1() -> list[dict]:
    """Gemini Thinking-Level Sweep — 9 trials."""
    console.print("\n[bold magenta]╔══════════════════════════════════════════╗[/bold magenta]")
    console.print("[bold magenta]║  EXPERIMENT 1: Gemini Thinking-Level Sweep  ║[/bold magenta]")
    console.print("[bold magenta]╚══════════════════════════════════════════╝[/bold magenta]")

    # (model_name, provider, thinking_level)
    trials_spec = [
        # gemini-3.1-pro-preview × low, medium, high
        (PRO,        GOOGLE, "low"),
        (PRO,        GOOGLE, "medium"),
        (PRO,        GOOGLE, "high"),
        # gemini-3-flash-preview × minimal, low, medium, high
        (FLASH,      GOOGLE, "minimal"),
        (FLASH,      GOOGLE, "low"),
        (FLASH,      GOOGLE, "medium"),
        (FLASH,      GOOGLE, "high"),
        # gemini-3.1-flash-lite-preview × minimal, low
        (FLASH_LITE, GOOGLE, "minimal"),
        (FLASH_LITE, GOOGLE, "low"),
    ]

    results: list[dict] = []
    total = len(trials_spec)

    for i, (model_name, provider, thinking_level) in enumerate(trials_spec, 1):
        label = f"gemini thinking_level={thinking_level} model={model_name}"
        _print_banner(i, total, label)

        gen_cfg = mc(provider, model_name, 0.3,  thinking_level)
        fin_cfg = mc(provider, model_name, 1.0,  thinking_level)
        adv_cfg = mc(provider, model_name, 1.0,  thinking_level)
        ref_cfg = mc(provider, model_name, 0.1,  thinking_level)

        row = {
            "model": model_name,
            "thinking_level": thinking_level,
            "finder_temp": 1.0,
            "error": "",
        }

        try:
            metrics = run_trial(gen_cfg, fin_cfg, adv_cfg, ref_cfg, TASK_A)
            row.update(metrics)
            console.print(
                f"[green]✓ Done:[/green] found={metrics['issues_found']} "
                f"confirmed={metrics['issues_confirmed']} "
                f"kills={metrics['adversary_kills']} ({metrics['kill_rate']}%) "
                f"prog={metrics['programmatic_findings']} "
                f"quality={metrics['code_quality_verdict']}"
            )
        except Exception as e:
            console.print(f"[red]✗ FAILED:[/red] {e}")
            traceback.print_exc()
            row.update(_err_row())
            row["error"] = str(e)[:200]

        results.append(row)

    # Save CSV
    csv_path = RESULTS_DIR / "thinking_level_sweep.csv"
    fieldnames = [
        "model", "thinking_level", "finder_temp",
        "issues_found", "issues_confirmed", "adversary_kills", "kill_rate",
        "programmatic_findings", "code_quality_verdict", "error",
    ]
    _write_csv(csv_path, results, fieldnames)

    # Print summary table
    table = Table(title="Experiment 1 — Gemini Thinking-Level Sweep (PostgreSQL SQL task)",
                  show_lines=True)
    table.add_column("Model", width=30)
    table.add_column("Thinking", width=9)
    table.add_column("Found", justify="right", width=6)
    table.add_column("Conf.", justify="right", width=6)
    table.add_column("Kills", justify="right", width=6)
    table.add_column("Kill%", justify="right", width=7)
    table.add_column("Prog.", justify="right", width=6)
    table.add_column("Quality", width=14)

    for r in results:
        qc = r.get("code_quality_verdict", "unknown")
        color = _quality_color(qc)
        table.add_row(
            r["model"], r["thinking_level"],
            str(r.get("issues_found", 0)), str(r.get("issues_confirmed", 0)),
            str(r.get("adversary_kills", 0)), f"{r.get('kill_rate', 0):.0f}%",
            str(r.get("programmatic_findings", 0)),
            f"[{color}]{qc}[/{color}]",
        )

    console.print()
    console.print(table)
    return results


# ── Experiment 2 ───────────────────────────────────────────────────────────────

def _exp2_configs() -> list[dict]:
    """Return the 6 cross-provider config definitions."""
    return [
        {
            "config_name": "config1_anthropic_finder_google_adversary",
            "generator": mc(GOOGLE,     FLASH,  0.3, "high"),
            "finder":    mc(ANTHROPIC,  SONNET, 0.7),
            "adversary": mc(GOOGLE,     FLASH,  1.0, "high"),
            "referee":   mc(ANTHROPIC,  SONNET, 0.1),
            "label": "Config 1 — Anthropic Finder, Google Adversary",
        },
        {
            "config_name": "config2_google_finder_anthropic_adversary",
            "generator": mc(GOOGLE,     FLASH,  0.3, "high"),
            "finder":    mc(GOOGLE,     FLASH,  1.0, "high"),
            "adversary": mc(ANTHROPIC,  SONNET, 0.7),
            "referee":   mc(ANTHROPIC,  SONNET, 0.1),
            "label": "Config 2 — Google Finder, Anthropic Adversary",
        },
        {
            "config_name": "config3_opus_finder_pro_adversary",
            "generator": mc(GOOGLE,     FLASH,  0.3, "high"),
            "finder":    mc(ANTHROPIC,  OPUS,   0.7),
            "adversary": mc(GOOGLE,     PRO,    1.0, "high"),
            "referee":   mc(GOOGLE,     PRO,    0.1, "high"),
            "label": "Config 3 — Opus Finder, Pro Adversary+Referee",
        },
        {
            "config_name": "config4_opus_finder_sonnet_adversary",
            "generator": mc(GOOGLE,     FLASH,  0.3, "high"),
            "finder":    mc(ANTHROPIC,  OPUS,   0.7),
            "adversary": mc(ANTHROPIC,  SONNET, 0.7),
            "referee":   mc(GOOGLE,     PRO,    0.1, "high"),
            "label": "Config 4 — Opus Finder, Sonnet Adversary, Pro Referee",
        },
        {
            "config_name": "config5_opus_finder_haiku_adversary",
            "generator": mc(GOOGLE,     FLASH,  0.3, "high"),
            "finder":    mc(ANTHROPIC,  OPUS,   0.7),
            "adversary": mc(ANTHROPIC,  HAIKU,  0.7),
            "referee":   mc(GOOGLE,     PRO,    0.1, "high"),
            "label": "Config 5 — Opus Finder, Haiku Adversary, Pro Referee",
        },
        {
            "config_name": "config6_all_different_google_heavy",
            "generator": mc(ANTHROPIC,  SONNET, 0.3),
            "finder":    mc(GOOGLE,     PRO,    1.0, "high"),
            "adversary": mc(GOOGLE,     FLASH,  1.0, "high"),
            "referee":   mc(ANTHROPIC,  OPUS,   0.1),
            "label": "Config 6 — Sonnet Gen, Pro Finder, Flash Adversary, Opus Referee",
        },
    ]


def run_experiment_2() -> list[dict]:
    """Cross-Provider Mixed-Model Trials — 12 trials (6 configs × 2 tasks)."""
    console.print("\n[bold magenta]╔═════════════════════════════════════════════════╗[/bold magenta]")
    console.print("[bold magenta]║  EXPERIMENT 2: Cross-Provider Mixed-Model Trials  ║[/bold magenta]")
    console.print("[bold magenta]╚═════════════════════════════════════════════════╝[/bold magenta]")

    configs = _exp2_configs()
    tasks = [("task_a_postgresql", TASK_A), ("task_b_ssrf_api", TASK_B)]

    results: list[dict] = []
    trial_num = 0
    total = len(configs) * len(tasks)

    for cfg in configs:
        for task_key, task_spec in tasks:
            trial_num += 1
            label = f"{cfg['label']} | {task_key}"
            _print_banner(trial_num, total, label)

            gen_cfg = cfg["generator"]
            fin_cfg = cfg["finder"]
            adv_cfg = cfg["adversary"]
            ref_cfg = cfg["referee"]

            row = {
                "config_name": cfg["config_name"],
                "task": task_key,
                "generator": gen_cfg.display_name,
                "finder": fin_cfg.display_name,
                "adversary": adv_cfg.display_name,
                "referee": ref_cfg.display_name,
                "finder_temp": fin_cfg.temperature,
                "error": "",
            }

            try:
                metrics = run_trial(gen_cfg, fin_cfg, adv_cfg, ref_cfg, task_spec)
                row.update(metrics)
                console.print(
                    f"[green]✓ Done:[/green] found={metrics['issues_found']} "
                    f"confirmed={metrics['issues_confirmed']} "
                    f"kills={metrics['adversary_kills']} ({metrics['kill_rate']}%) "
                    f"prog={metrics['programmatic_findings']} "
                    f"quality={metrics['code_quality_verdict']}"
                )
            except Exception as e:
                console.print(f"[red]✗ FAILED:[/red] {e}")
                traceback.print_exc()
                row.update(_err_row())
                row["error"] = str(e)[:200]

            results.append(row)

    # Save CSV
    csv_path = RESULTS_DIR / "mixed_model_comparison.csv"
    fieldnames = [
        "config_name", "task", "generator", "finder", "adversary", "referee",
        "finder_temp", "issues_found", "issues_confirmed", "adversary_kills",
        "kill_rate", "programmatic_findings", "code_quality_verdict", "error",
    ]
    _write_csv(csv_path, results, fieldnames)

    # Print summary table
    table = Table(title="Experiment 2 — Cross-Provider Mixed-Model Trials",
                  show_lines=True)
    table.add_column("Config", width=38)
    table.add_column("Task", width=18)
    table.add_column("Found", justify="right", width=6)
    table.add_column("Conf.", justify="right", width=6)
    table.add_column("Kills", justify="right", width=6)
    table.add_column("Kill%", justify="right", width=7)
    table.add_column("Prog.", justify="right", width=6)
    table.add_column("Quality", width=14)

    for r in results:
        qc = r.get("code_quality_verdict", "unknown")
        color = _quality_color(qc)
        table.add_row(
            r["config_name"], r["task"],
            str(r.get("issues_found", 0)), str(r.get("issues_confirmed", 0)),
            str(r.get("adversary_kills", 0)), f"{r.get('kill_rate', 0):.0f}%",
            str(r.get("programmatic_findings", 0)),
            f"[{color}]{qc}[/{color}]",
        )

    console.print()
    console.print(table)
    return results


# ── Experiment 3 ───────────────────────────────────────────────────────────────

def run_experiment_3() -> list[dict]:
    """Sonnet Self-Review Temperature Threshold — 6 trials."""
    console.print("\n[bold magenta]╔══════════════════════════════════════════════════════╗[/bold magenta]")
    console.print("[bold magenta]║  EXPERIMENT 3: Sonnet Self-Review Temperature Threshold  ║[/bold magenta]")
    console.print("[bold magenta]╚══════════════════════════════════════════════════════╝[/bold magenta]")

    finder_temps = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    results: list[dict] = []
    total = len(finder_temps)

    for i, finder_temp in enumerate(finder_temps, 1):
        label = f"Sonnet self-review finder_temp={finder_temp}"
        _print_banner(i, total, label)

        gen_cfg = mc(ANTHROPIC, SONNET, 0.3)
        fin_cfg = mc(ANTHROPIC, SONNET, finder_temp)
        adv_cfg = mc(ANTHROPIC, SONNET, 0.7)
        ref_cfg = mc(ANTHROPIC, SONNET, 0.1)

        row = {"finder_temp": finder_temp, "error": ""}

        try:
            metrics = run_trial(gen_cfg, fin_cfg, adv_cfg, ref_cfg, TASK_A)
            row.update(metrics)
            console.print(
                f"[green]✓ Done:[/green] found={metrics['issues_found']} "
                f"confirmed={metrics['issues_confirmed']} "
                f"kill_rate={metrics['kill_rate']}% "
                f"quality={metrics['code_quality_verdict']}"
            )
        except Exception as e:
            console.print(f"[red]✗ FAILED:[/red] {e}")
            traceback.print_exc()
            row.update(_err_row())
            row["error"] = str(e)[:200]

        results.append(row)

    # Save CSV
    csv_path = RESULTS_DIR / "sonnet_self_review_threshold.csv"
    fieldnames = [
        "finder_temp", "issues_found", "issues_confirmed",
        "adversary_kills", "kill_rate", "programmatic_findings",
        "code_quality_verdict", "error",
    ]
    _write_csv(csv_path, results, fieldnames)

    # Print summary table
    table = Table(title="Experiment 3 — Sonnet Self-Review Temperature Threshold",
                  show_lines=True)
    table.add_column("Finder Temp", justify="right", width=12)
    table.add_column("Found", justify="right", width=6)
    table.add_column("Conf.", justify="right", width=6)
    table.add_column("Kills", justify="right", width=6)
    table.add_column("Kill%", justify="right", width=7)
    table.add_column("Prog.", justify="right", width=6)
    table.add_column("Quality", width=14)

    for r in results:
        qc = r.get("code_quality_verdict", "unknown")
        color = _quality_color(qc)
        table.add_row(
            f"{r['finder_temp']:.1f}",
            str(r.get("issues_found", 0)), str(r.get("issues_confirmed", 0)),
            str(r.get("adversary_kills", 0)), f"{r.get('kill_rate', 0):.0f}%",
            str(r.get("programmatic_findings", 0)),
            f"[{color}]{qc}[/{color}]",
        )

    console.print()
    console.print(table)
    return results


# ── Utilities ──────────────────────────────────────────────────────────────────

def _quality_color(qc: str) -> str:
    return {
        "secure": "green", "mostly_secure": "green",
        "concerning": "yellow", "vulnerable": "red",
        "critical": "bold red", "error": "dim red",
    }.get(qc, "white")


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)
    console.print(f"[green]CSV saved:[/green] {path}")


# ── Master Summary ─────────────────────────────────────────────────────────────

def print_master_summary(
    exp1: list[dict],
    exp2: list[dict],
    exp3: list[dict],
) -> None:
    console.print("\n\n")
    console.print("[bold underline white on blue]" + "═" * 72 + "[/bold underline white on blue]")
    console.print("[bold underline white on blue]  MASTER SUMMARY — April 7, 2026 MAAT Experiment Batch" +
                  " " * 19 + "[/bold underline white on blue]")
    console.print("[bold underline white on blue]" + "═" * 72 + "[/bold underline white on blue]")

    # ── Q1: Optimal thinking_level per Gemini model ──
    console.print("\n[bold yellow]Q1: Optimal thinking_level for each Gemini model as Finder[/bold yellow]")
    models_seen = {}
    for r in exp1:
        m = r.get("model", "")
        if r.get("error"):
            continue
        if m not in models_seen:
            models_seen[m] = []
        models_seen[m].append(r)

    for model_name, rows in models_seen.items():
        if not rows:
            console.print(f"  {model_name}: no successful trials")
            continue
        best = max(rows, key=lambda x: (x.get("issues_confirmed", 0), x.get("issues_found", 0)))
        console.print(
            f"  [cyan]{model_name}[/cyan]: best = thinking_level=[bold]{best['thinking_level']}[/bold] "
            f"→ confirmed={best['issues_confirmed']} / found={best['issues_found']}"
        )

    # ── Q2: Best cross-provider config ──
    console.print("\n[bold yellow]Q2: Cross-provider config with most confirmed real issues[/bold yellow]")
    valid_exp2 = [r for r in exp2 if not r.get("error")]
    if valid_exp2:
        # Sum confirmed across both tasks per config
        config_totals: dict[str, dict] = {}
        for r in valid_exp2:
            cn = r["config_name"]
            if cn not in config_totals:
                config_totals[cn] = {"confirmed": 0, "found": 0, "rows": []}
            config_totals[cn]["confirmed"] += r.get("issues_confirmed", 0)
            config_totals[cn]["found"] += r.get("issues_found", 0)
            config_totals[cn]["rows"].append(r)

        best_cn = max(config_totals, key=lambda c: config_totals[c]["confirmed"])
        bd = config_totals[best_cn]
        console.print(
            f"  Best config: [bold cyan]{best_cn}[/bold cyan]\n"
            f"  Total confirmed across both tasks: [bold green]{bd['confirmed']}[/bold green] "
            f"(out of {bd['found']} found)"
        )
        console.print("  Per-task breakdown:")
        for r in bd["rows"]:
            console.print(
                f"    {r['task']}: found={r.get('issues_found',0)} "
                f"confirmed={r.get('issues_confirmed',0)} quality={r.get('code_quality_verdict','?')}"
            )
    else:
        console.print("  No successful Exp 2 trials.")

    # ── Q3: Cross-provider adversary quality ──
    console.print("\n[bold yellow]Q3: Does a cross-provider adversary produce better results?[/bold yellow]")
    # Config 1 (Anthropic finder, Google adversary) vs Config 2 (Google finder, Anthropic adversary)
    # Compare vs Config 3/4/5 where Opus finder uses different adversary providers
    cross_configs = [r for r in valid_exp2 if "config1" in r.get("config_name","") or
                                               "config2" in r.get("config_name","")]
    same_ish = [r for r in valid_exp2 if "config3" in r.get("config_name","") or
                                          "config4" in r.get("config_name","") or
                                          "config5" in r.get("config_name","")]
    if cross_configs and same_ish:
        avg_cross_confirmed = sum(r.get("issues_confirmed",0) for r in cross_configs) / len(cross_configs)
        avg_same_confirmed  = sum(r.get("issues_confirmed",0) for r in same_ish)  / len(same_ish)
        avg_cross_kill = sum(r.get("kill_rate",0) for r in cross_configs) / len(cross_configs)
        avg_same_kill  = sum(r.get("kill_rate",0) for r in same_ish)  / len(same_ish)
        console.print(
            f"  Mixed-provider configs (1&2) avg confirmed: {avg_cross_confirmed:.1f}  "
            f"avg kill_rate: {avg_cross_kill:.1f}%"
        )
        console.print(
            f"  Opus-finder configs (3-5) avg confirmed: {avg_same_confirmed:.1f}  "
            f"avg kill_rate: {avg_same_kill:.1f}%"
        )
        if avg_cross_confirmed > avg_same_confirmed:
            console.print("  → Cross-provider configs produced MORE confirmed issues on average.")
        elif avg_cross_confirmed < avg_same_confirmed:
            console.print("  → Opus-finder configs produced MORE confirmed issues on average.")
        else:
            console.print("  → Results are even between configurations.")
    else:
        console.print("  Insufficient data for comparison.")

    # ── Q4: Sonnet temperature threshold ──
    console.print("\n[bold yellow]Q4: Sonnet self-review leniency threshold[/bold yellow]")
    valid_exp3 = [r for r in exp3 if not r.get("error")]
    if valid_exp3:
        # Find the temperature where confirmed issues jumps meaningfully
        # Use first point where confirmed > 0 or where there's a notable increase
        baseline_confirmed = valid_exp3[0].get("issues_confirmed", 0) if valid_exp3 else 0
        threshold_temp = None
        for r in valid_exp3:
            confirmed = r.get("issues_confirmed", 0)
            if confirmed > baseline_confirmed:
                threshold_temp = r["finder_temp"]
                baseline_confirmed = confirmed
                break  # first breakout point

        # Also look for max confirmed
        best_exp3 = max(valid_exp3, key=lambda x: (x.get("issues_confirmed",0), x.get("issues_found",0)))
        console.print("  Temp → (found, confirmed, kill_rate%):")
        for r in valid_exp3:
            marker = " ← breakout" if r["finder_temp"] == threshold_temp else ""
            marker2 = " ← peak" if r["finder_temp"] == best_exp3["finder_temp"] and not marker else ""
            console.print(
                f"    {r['finder_temp']:.1f}: found={r.get('issues_found',0)} "
                f"confirmed={r.get('issues_confirmed',0)} "
                f"kill={r.get('kill_rate',0):.0f}%{marker}{marker2}"
            )
        if threshold_temp:
            console.print(f"\n  [bold green]Threshold: finder_temp ≥ {threshold_temp:.1f}[/bold green]")
        else:
            console.print("  [yellow]No clear breakout detected — self-review leniency persisted across all temps.[/yellow]")
    else:
        console.print("  No successful Exp 3 trials.")

    # ── Q5: Best overall config ──
    console.print("\n[bold yellow]Q5: Single best configuration for maximizing confirmed real vulnerabilities[/bold yellow]")

    # Gather all trials with metrics
    all_rows: list[dict] = []

    for r in exp1:
        if not r.get("error"):
            all_rows.append({
                "label": f"Exp1: {r['model']} thinking={r['thinking_level']}",
                "confirmed": r.get("issues_confirmed", 0),
                "found": r.get("issues_found", 0),
                "prog": r.get("programmatic_findings", 0),
            })

    for r in exp2:
        if not r.get("error"):
            all_rows.append({
                "label": f"Exp2: {r['config_name']} | {r['task']}",
                "confirmed": r.get("issues_confirmed", 0),
                "found": r.get("issues_found", 0),
                "prog": r.get("programmatic_findings", 0),
            })

    for r in exp3:
        if not r.get("error"):
            all_rows.append({
                "label": f"Exp3: Sonnet self-review finder_temp={r['finder_temp']}",
                "confirmed": r.get("issues_confirmed", 0),
                "found": r.get("issues_found", 0),
                "prog": r.get("programmatic_findings", 0),
            })

    if all_rows:
        # Score = confirmed + 0.2 * prog (programmatic findings add secondary signal)
        best_overall = max(all_rows, key=lambda x: x["confirmed"] + 0.2 * x["prog"])
        console.print(
            f"\n  [bold green]Best overall:[/bold green] {best_overall['label']}\n"
            f"  confirmed={best_overall['confirmed']} found={best_overall['found']} "
            f"prog={best_overall['prog']}"
        )

        # Top 3
        top3 = sorted(all_rows, key=lambda x: x["confirmed"] + 0.2*x["prog"], reverse=True)[:3]
        console.print("\n  Top 3 configurations by confirmed issues:")
        for rank, row in enumerate(top3, 1):
            console.print(f"    #{rank}: confirmed={row['confirmed']} prog={row['prog']} → {row['label']}")
    else:
        console.print("  Insufficient data.")

    console.print("\n[bold white]═" * 72 + "[/bold white]")
    console.print("[bold white]  End of April 7 Batch Report[/bold white]")
    console.print("[bold white]═" * 72 + "[/bold white]\n")


# ── Entry Point ────────────────────────────────────────────────────────────────

def main() -> None:
    console.print("[bold white on dark_green]" + " " * 72 + "[/bold white on dark_green]")
    console.print("[bold white on dark_green]  MAAT — April 7, 2026 Experiment Batch" +
                  " " * 34 + "[/bold white on dark_green]")
    console.print("[bold white on dark_green]  27 trials: Exp1 (9) + Exp2 (12) + Exp3 (6)" +
                  " " * 27 + "[/bold white on dark_green]")
    console.print("[bold white on dark_green]" + " " * 72 + "[/bold white on dark_green]")

    exp1_results = run_experiment_1()
    exp2_results = run_experiment_2()
    exp3_results = run_experiment_3()

    print_master_summary(exp1_results, exp2_results, exp3_results)


if __name__ == "__main__":
    main()
