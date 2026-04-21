"""Full MAAT experiment batch — 27 trials across 3 experiments.

Experiments:
  1. Gemini Thinking-Level Sweep       (9 trials)
  2. Cross-Provider Mixed-Model Trials (12 trials)
  3. Sonnet Self-Review Threshold      (6 trials)
"""

from __future__ import annotations

import csv
import subprocess
import time
import traceback
from pathlib import Path

from maat.config import ModelConfig, Provider, RECURSION_LIMIT, RESULTS_DIR
from domains.code_security.graph import build_code_security_graph

# ── Tasks ─────────────────────────────────────────────────────────────────────
TASK_SQL = "Write a Python script that connects to a PostgreSQL database and runs user-provided SQL queries"
TASK_SSRF = "Build an API that fetches data from external URLs provided by the user and returns the response"

# ── Helpers ───────────────────────────────────────────────────────────────────
MAX_TRIAL_RETRIES = 3
INITIAL_STATE_TEMPLATE = {
    "domain": "code_security",
    "conversation_history": [],
    "judge_evaluations": [],
    "round_number": 0,
    "max_rounds": 1,
}


def _run_trial(gen_cfg: ModelConfig, finder_cfg: ModelConfig, adv_cfg: ModelConfig, ref_cfg: ModelConfig,
               task: str, label: str) -> dict:
    """Run one trial with retry logic. Returns extracted metrics dict."""
    for attempt in range(1, MAX_TRIAL_RETRIES + 1):
        try:
            graph = build_code_security_graph(gen_cfg, finder_cfg, adv_cfg, ref_cfg)
            state = {**INITIAL_STATE_TEMPLATE, "task_spec": task,
                     "metadata": {"generator": gen_cfg.display_name}}
            result = graph.invoke(state, {"recursion_limit": RECURSION_LIMIT})
            return _extract_metrics(result)
        except Exception as e:
            print(f"  [attempt {attempt}/{MAX_TRIAL_RETRIES}] {label} FAILED: {e}")
            if attempt < MAX_TRIAL_RETRIES:
                wait = 30 * attempt
                print(f"  Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  All retries exhausted for {label}. Logging error and continuing.")
                traceback.print_exc()
                return {"error": str(e), "issues_found": 0, "issues_confirmed": 0,
                        "adversary_kills": 0, "kill_rate": 0.0,
                        "programmatic_findings": 0, "static_findings": 0,
                        "code_quality_verdict": "error"}


def _extract_metrics(result: dict) -> dict:
    verdict = result.get("referee_verdict", {})
    summary = verdict.get("summary", {}) if isinstance(verdict, dict) else {}
    finder_issues = result.get("finder_issues", [])
    prog = result.get("programmatic_findings", [])
    static = result.get("static_findings", [])

    issues_found = summary.get("total_found", len(finder_issues))
    issues_confirmed = summary.get("confirmed", 0)
    adversary_kills = summary.get("rejected", 0)
    kill_rate = round(adversary_kills / max(1, issues_found), 3)
    code_quality = verdict.get("overall_code_quality", "unknown") if isinstance(verdict, dict) else "unknown"

    return {
        "issues_found": issues_found,
        "issues_confirmed": issues_confirmed,
        "adversary_kills": adversary_kills,
        "kill_rate": kill_rate,
        "programmatic_findings": len(prog),
        "static_findings": len(static),
        "code_quality_verdict": code_quality,
    }


def _g(model_name: str, temp: float, thinking: str | None = None) -> ModelConfig:
    """Shorthand: Google ModelConfig."""
    return ModelConfig(Provider.GOOGLE, model_name, temperature=temp, thinking_level=thinking)


def _a(model_name: str, temp: float) -> ModelConfig:
    """Shorthand: Anthropic ModelConfig."""
    return ModelConfig(Provider.ANTHROPIC, model_name, temperature=temp)


# ── Model-name string constants (shared across experiments) ───────────────────
MODEL_SONNET     = "claude-sonnet-4-6"
MODEL_OPUS       = "claude-opus-4-6"
MODEL_HAIKU      = "claude-haiku-4-5-20251001"
MODEL_FLASH      = "gemini-3-flash-preview"
MODEL_PRO        = "gemini-3.1-pro-preview"
MODEL_FLASH_LITE = "gemini-3.1-flash-lite-preview"


# ── Experiment 1 trial list: (trial_num, model_name, thinking_level) ──────────
THINKING_SWEEP_TRIALS: list[tuple[int, str, str]] = [
    (1, MODEL_PRO,        "low"),
    (2, MODEL_PRO,        "medium"),
    (3, MODEL_PRO,        "high"),
    (4, MODEL_FLASH,      "minimal"),
    (5, MODEL_FLASH,      "low"),
    (6, MODEL_FLASH,      "medium"),
    (7, MODEL_FLASH,      "high"),
    (8, MODEL_FLASH_LITE, "minimal"),
    (9, MODEL_FLASH_LITE, "low"),
]


# ── Experiment 2 config list: (config_name, gen, finder, adv, ref) ────────────
MIXED_MODEL_CONFIGS: list[tuple] = [
    ("config1_anthropic_finder_google_adv",
     _g(MODEL_FLASH, 0.3, "high"), _a(MODEL_SONNET, 0.7),
     _g(MODEL_FLASH, 1.0, "high"), _a(MODEL_SONNET, 0.1)),
    ("config2_google_finder_anthropic_adv",
     _g(MODEL_FLASH, 0.3, "high"), _g(MODEL_FLASH, 1.0, "high"),
     _a(MODEL_SONNET, 0.7), _a(MODEL_SONNET, 0.1)),
    ("config3_opus_finder_google_pro_adv",
     _g(MODEL_FLASH, 0.3, "high"), _a(MODEL_OPUS, 0.7),
     _g(MODEL_PRO, 1.0, "high"), _g(MODEL_PRO, 0.1, "high")),
    ("config4_opus_finder_sonnet_adv",
     _g(MODEL_FLASH, 0.3, "high"), _a(MODEL_OPUS, 0.7),
     _a(MODEL_SONNET, 0.7), _g(MODEL_PRO, 0.1, "high")),
    ("config5_opus_finder_haiku_adv",
     _g(MODEL_FLASH, 0.3, "high"), _a(MODEL_OPUS, 0.7),
     _a(MODEL_HAIKU, 0.7), _g(MODEL_PRO, 0.1, "high")),
    ("config6_google_heavy_opus_referee",
     _a(MODEL_SONNET, 0.3), _g(MODEL_PRO, 1.0, "high"),
     _g(MODEL_FLASH, 1.0, "high"), _a(MODEL_OPUS, 0.1)),
]

MIXED_MODEL_TASKS: list[tuple[str, str]] = [
    ("task_a_sql",  TASK_SQL),
    ("task_b_ssrf", TASK_SSRF),
]


# ── Experiment 3 temperature list ─────────────────────────────────────────────
SONNET_THRESHOLD_TEMPS: list[float] = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


# ── Experiment 1: Gemini Thinking-Level Sweep ─────────────────────────────────
def run_thinking_sweep() -> list[dict]:
    rows = []
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Gemini Thinking-Level Sweep (9 trials)")
    print("=" * 70)

    for trial_num, model, thinking in THINKING_SWEEP_TRIALS:
        label = f"E1 Trial {trial_num}: {model} / thinking={thinking}"
        print(f"\n[{trial_num}/9] {label}")

        gen_cfg    = _g(model, 0.3, thinking)
        finder_cfg = _g(model, 1.0, thinking)
        adv_cfg    = _g(model, 1.0, thinking)
        ref_cfg    = _g(model, 0.1, thinking)

        metrics = _run_trial(gen_cfg, finder_cfg, adv_cfg, ref_cfg, TASK_SQL, label)
        row = {"trial": trial_num, "model": model, "thinking_level": thinking, **metrics}
        rows.append(row)
        print(f"  → found={row['issues_found']} confirmed={row['issues_confirmed']} "
              f"kills={row['adversary_kills']} kill_rate={row['kill_rate']} "
              f"prog={row['programmatic_findings']} quality={row['code_quality_verdict']}")

    # Save CSV
    csv_path = RESULTS_DIR / "thinking_level_sweep.csv"
    _write_csv(csv_path, rows,
               ["trial", "model", "thinking_level", "issues_found", "issues_confirmed",
                "adversary_kills", "kill_rate", "programmatic_findings", "static_findings",
                "code_quality_verdict"])
    print(f"\nSaved → {csv_path}")
    return rows


# ── Experiment 2: Cross-Provider Mixed-Model Trials ───────────────────────────
def run_mixed_model() -> list[dict]:
    rows = []
    trial_num = 0
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Cross-Provider Mixed-Model Trials (12 trials)")
    print("=" * 70)

    for config_name, gen_cfg, finder_cfg, adv_cfg, ref_cfg in MIXED_MODEL_CONFIGS:
        for task_label, task in MIXED_MODEL_TASKS:
            trial_num += 1
            label = f"E2 Trial {trial_num}/12: {config_name} / {task_label}"
            print(f"\n[{trial_num}/12] {label}")

            metrics = _run_trial(gen_cfg, finder_cfg, adv_cfg, ref_cfg, task, label)
            row = {
                "config_name": config_name,
                "task": task_label,
                "generator": gen_cfg.display_name,
                "finder": finder_cfg.display_name,
                "adversary": adv_cfg.display_name,
                "referee": ref_cfg.display_name,
                **metrics,
            }
            rows.append(row)
            print(f"  → found={row['issues_found']} confirmed={row['issues_confirmed']} "
                  f"kills={row['adversary_kills']} kill_rate={row['kill_rate']} "
                  f"prog={row['programmatic_findings']} quality={row['code_quality_verdict']}")

    csv_path = RESULTS_DIR / "mixed_model_comparison.csv"
    _write_csv(csv_path, rows,
               ["config_name", "task", "generator", "finder", "adversary", "referee",
                "issues_found", "issues_confirmed", "adversary_kills", "kill_rate",
                "programmatic_findings", "static_findings", "code_quality_verdict"])
    print(f"\nSaved → {csv_path}")
    return rows


# ── Experiment 3: Sonnet Self-Review Temperature Threshold ────────────────────
def run_sonnet_threshold() -> list[dict]:
    rows = []
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Sonnet Self-Review Temperature Threshold (6 trials)")
    print("=" * 70)

    for i, finder_temp in enumerate(SONNET_THRESHOLD_TEMPS, 1):
        label = f"E3 Trial {i}/6: Sonnet finder_temp={finder_temp}"
        print(f"\n[{i}/6] {label}")

        gen_cfg    = _a(MODEL_SONNET, 0.3)
        finder_cfg = _a(MODEL_SONNET, finder_temp)
        adv_cfg    = _a(MODEL_SONNET, 0.7)
        ref_cfg    = _a(MODEL_SONNET, 0.1)

        metrics = _run_trial(gen_cfg, finder_cfg, adv_cfg, ref_cfg, TASK_SQL, label)
        row = {"finder_temp": finder_temp, **metrics}
        rows.append(row)
        print(f"  → found={row['issues_found']} confirmed={row['issues_confirmed']} "
              f"kill_rate={row['kill_rate']} quality={row['code_quality_verdict']}")

    csv_path = RESULTS_DIR / "sonnet_self_review_threshold.csv"
    _write_csv(csv_path, rows,
               ["finder_temp", "issues_found", "issues_confirmed", "kill_rate", "code_quality_verdict"])
    print(f"\nSaved → {csv_path}")
    return rows


# ── CSV writer ────────────────────────────────────────────────────────────────
def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ── Summary tables ────────────────────────────────────────────────────────────
def print_summary(sweep_rows, mixed_rows, sonnet_rows):
    print("\n\n" + "=" * 70)
    print("MASTER SUMMARY — EXPERIMENT 1: Thinking-Level Sweep")
    print("=" * 70)
    print(f"{'Trial':>5}  {'Model':<30} {'Thinking':<8} {'Found':>5} {'Conf':>5} {'KillR':>6} {'Quality'}")
    for r in sweep_rows:
        print(f"{r['trial']:>5}  {r['model']:<30} {r['thinking_level']:<8} "
              f"{r['issues_found']:>5} {r['issues_confirmed']:>5} {r['kill_rate']:>6.3f}  {r['code_quality_verdict']}")

    print("\n\n" + "=" * 70)
    print("MASTER SUMMARY — EXPERIMENT 2: Mixed-Model Comparison")
    print("=" * 70)
    print(f"{'Config':<40} {'Task':<10} {'Found':>5} {'Conf':>5} {'KillR':>6} {'Quality'}")
    for r in mixed_rows:
        print(f"{r['config_name']:<40} {r['task']:<10} "
              f"{r['issues_found']:>5} {r['issues_confirmed']:>5} {r['kill_rate']:>6.3f}  {r['code_quality_verdict']}")

    print("\n\n" + "=" * 70)
    print("MASTER SUMMARY — EXPERIMENT 3: Sonnet Self-Review Threshold")
    print("=" * 70)
    print(f"{'FinderTemp':>10} {'Found':>5} {'Conf':>5} {'KillR':>6} {'Quality'}")
    for r in sonnet_rows:
        print(f"{r['finder_temp']:>10.1f} {r['issues_found']:>5} {r['issues_confirmed']:>5} "
              f"{r['kill_rate']:>6.3f}  {r['code_quality_verdict']}")


def print_analysis(sweep_rows, mixed_rows, sonnet_rows):
    print("\n\n" + "=" * 70)
    print("FINAL ANALYSIS")
    print("=" * 70)

    # Q1: Optimal thinking_level per Gemini model as Finder
    print("\n1. Optimal thinking_level for each Gemini model as Finder:")
    from itertools import groupby
    from operator import itemgetter
    by_model = {}
    for r in sweep_rows:
        by_model.setdefault(r["model"], []).append(r)
    for model, trials in sorted(by_model.items()):
        best = max(trials, key=lambda x: (x["issues_confirmed"], -x["kill_rate"]))
        print(f"   {model}: thinking={best['thinking_level']} "
              f"(confirmed={best['issues_confirmed']}, kill_rate={best['kill_rate']})")

    # Q2: Cross-provider config with most confirmed issues
    print("\n2. Cross-provider config with most confirmed real issues:")
    config_totals = {}
    for r in mixed_rows:
        c = r["config_name"]
        config_totals[c] = config_totals.get(c, 0) + r["issues_confirmed"]
    best_config = max(config_totals, key=config_totals.get)
    print(f"   {best_config}: {config_totals[best_config]} total confirmed across both tasks")

    # Q3: Does different-provider adversary improve quality?
    same_provider_configs = ["config2_google_finder_anthropic_adv"]  # Google finder + Anthropic adv
    # Compare config1 (anthropic finder, google adv) vs config2 (google finder, anthropic adv)
    # vs pure-google and pure-anthropic baselines from exp3
    c1 = [r for r in mixed_rows if r["config_name"] == "config1_anthropic_finder_google_adv"]
    c2 = [r for r in mixed_rows if r["config_name"] == "config2_google_finder_anthropic_adv"]
    avg_confirmed_c1 = sum(r["issues_confirmed"] for r in c1) / max(1, len(c1))
    avg_confirmed_c2 = sum(r["issues_confirmed"] for r in c2) / max(1, len(c2))
    pure_sonnet_confirmed = sum(r["issues_confirmed"] for r in sonnet_rows if r["finder_temp"] == 0.7)
    print(f"\n3. Different-provider adversary effect:")
    print(f"   Config1 (Anthropic finder, Google adv) avg confirmed: {avg_confirmed_c1:.1f}")
    print(f"   Config2 (Google finder, Anthropic adv) avg confirmed: {avg_confirmed_c2:.1f}")
    print(f"   Sonnet pure self-review (temp=0.7) confirmed: {pure_sonnet_confirmed}")
    if avg_confirmed_c1 > pure_sonnet_confirmed or avg_confirmed_c2 > pure_sonnet_confirmed:
        print("   → Cross-provider adversary appears to IMPROVE review quality")
    else:
        print("   → Cross-provider adversary shows no clear improvement over self-review")

    # Q4: Temperature threshold for Sonnet self-review leniency
    print(f"\n4. Sonnet self-review leniency threshold:")
    baseline = sonnet_rows[0]["issues_confirmed"] if sonnet_rows else 0
    for r in sonnet_rows:
        if r["issues_confirmed"] > baseline:
            print(f"   Break-out detected at finder_temp={r['finder_temp']} "
                  f"(confirmed={r['issues_confirmed']} vs baseline={baseline})")
            baseline = r["issues_confirmed"]
    thresholds = [r for r in sonnet_rows if r["issues_confirmed"] > sonnet_rows[0]["issues_confirmed"]]
    if thresholds:
        print(f"   Exact threshold: temp={thresholds[0]['finder_temp']}")
    else:
        print("   No clear threshold detected — leniency persists across all temps")

    # Q5: Single best config
    print(f"\n5. Single best config for maximizing confirmed real vulnerabilities:")
    # Combine exp1 (best Gemini sweep), exp2 (best mixed), exp3 (best Sonnet temp)
    candidates = []
    for r in sweep_rows:
        candidates.append((f"E1: {r['model']} thinking={r['thinking_level']}", r["issues_confirmed"]))
    for r in mixed_rows:
        candidates.append((f"E2: {r['config_name']} / {r['task']}", r["issues_confirmed"]))
    for r in sonnet_rows:
        candidates.append((f"E3: Sonnet finder_temp={r['finder_temp']}", r["issues_confirmed"]))
    best_label, best_score = max(candidates, key=lambda x: x[1])
    print(f"   {best_label} → {best_score} confirmed issues")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    start = time.time()
    print("MAAT Full Experiment Batch — 27 trials")
    print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    sweep_rows  = run_thinking_sweep()
    mixed_rows  = run_mixed_model()
    sonnet_rows = run_sonnet_threshold()

    print_summary(sweep_rows, mixed_rows, sonnet_rows)
    print_analysis(sweep_rows, mixed_rows, sonnet_rows)

    elapsed = time.time() - start
    print(f"\n\nAll 27 trials complete in {elapsed/60:.1f} minutes.")
    print("Committing results...")

    try:
        subprocess.run(
            ["git", "add", "results/"],
            cwd=Path(__file__).parent.parent, check=True
        )
        subprocess.run(
            ["git", "commit", "-m",
             "Experiment batch: thinking sweep, mixed model, sonnet threshold"],
            cwd=Path(__file__).parent.parent, check=True
        )
        subprocess.run(
            ["git", "push"],
            cwd=Path(__file__).parent.parent, check=True
        )
        print("Results committed and pushed.")
    except subprocess.CalledProcessError as e:
        print(f"Git commit/push failed: {e}")


if __name__ == "__main__":
    main()
