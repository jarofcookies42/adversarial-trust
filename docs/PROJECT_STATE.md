# Project State

Last updated: 2026-04-20.

## Where things stand

**Adversarial-trust repo (Domain 2 — code security):**

- Migration: uv-managed Python 3.13 (.venv), pyproject.toml + uv.lock,
  dev tools in [project.optional-dependencies.dev], bandit in main
  dependencies. The old broken Homebrew Python 3.14 venv is gone.
  Archive of the April 7 batch environment is at
  archive/requirements-april7-snapshot.txt.

- Three-layer evaluation architecture for code_security domain:
  - Layer 1: regex detector (domains/code_security/detector.py) —
    unchanged from pilot, ~20 rules for SQL injection, eval/exec,
    weak crypto, hardcoded secrets, etc.
  - Layer 2: bandit static analysis (domains/code_security/static_scanner.py)
    — wraps bandit subprocess, returns findings in detector.Finding shape.
  - Layer 3: LLM referee (existing, in graph.py).
  - Generalized verdict escalation in graph.py: critical findings from
    either deterministic layer that the referee missed get appended to
    final_issues with source provenance, and overall_code_quality is
    lifted to at least "vulnerable". Detector findings and static
    findings are stored as SEPARATE fields in the result JSON
    (programmatic_findings vs static_findings) — do not merge.

- April 7 batch backfilled: 28 result JSONs now carry static_findings,
  three CSVs regenerated against trial definitions imported from
  full_experiment_batch.py (no schema duplication). All 28 JSONs are
  first-class tracked via a targeted .gitignore exception
  (!results/code_security_*_20260407_*.json).

- Test coverage: 73 passing tests. test_static_scanner.py (36) and
  test_escalation.py (34) cover the new modules. One pre-existing
  failure: test_temperature_conventions expects TEMP_ATTACKER == 0.9
  but live value is 0.7 with a comment in maat/config.py:16 explaining
  the deliberate change. Test wasn't updated; one-line fix when convenient.

**Cs5374project repo (Domain 1 — secret extraction, the pilot):**
- Untouched in this work. Working venv on Python 3.13. Same
  Homebrew-Python brittleness applies but it hasn't broken yet.
  Migrating to uv at some point is reasonable but not urgent.
- Contains the canonical experimental data and logs that ground the
  thesis claims (planet→ocean substitution, CRYSTAL flinch,
  capability-floor finding, etc.). Treat as read-only from
  adversarial-trust's perspective.

## The framing that matters

Domain 2's contribution is NOT "we built a multi-layer architecture
for adversarial code review." That's a crowded research area
(MAD-Spear, Free-MAD, various MAD frameworks; see April 2026
literature). The defensible thesis claim is:

> We measured the gap between syntactic and semantic defense in
> AI-generated code. Static analysis tools (regex, bandit) fire
> sparsely on modern frontier-model output because models have been
> effectively trained against the same SAST rubrics defenders use.
> The LLM referee carries the semantic load — particularly for
> patterns like user_input → cursor.execute(query, params) where the
> vulnerability is in dataflow rather than syntax.

Phase 3 backfill on the April 7 batch confirmed this: 4 static hits
across 27 trials, zero overlap between regex and bandit layers, LLM
catches things both static layers miss.

This reframe is more defensible than the "three-layer architecture
gives high coverage" story because it doesn't require you to compete
with a year of MAD literature, and the data supports it cleanly.

## What got considered and rejected (don't re-propose)

- Scoring mechanism / scoreboard for finder-adversary "competition":
  Rejected. Without a feedback loop between agents (which the current
  graph doesn't have — agents respond independently before any score
  exists), a scoreboard is decorative, not behavioral. Real incentive
  dynamics would require iterative debate or multi-round runs feeding
  prior scores back into prompts. Future architectural extension if
  Domain 2 becomes its own paper, not a near-term task.

- README's "+1 per issue / +1 per kill / +1 accuracy" framing:
  Removed. There's no scoring in the code. The prompts use "rewarded
  for" language as role-conditioning prose; that's a real technique
  but not engineered game theory. Don't reintroduce.

- Path A for static analysis (keep regex layer narrow, accept thin
  coverage, frame as "safety net for known footguns"): Considered and
  rejected in favor of Path B (add bandit). Path B was right — the
  layers genuinely capture different slices.

- Pinning April 7 dependency versions for "reproducibility": Considered
  and rejected. The actual ground truth is the saved JSONs in results/.
  Server-side model APIs have already drifted in 12 days regardless of
  client SDK version. Pinned anthropic 0.89 wouldn't reproduce April 7
  Sonnet behavior, just April 7 client behavior.

- Claude Code suggestion to "improve" the regex detector by folding it
  into bandit: Rejected. Layers must remain measurable separately for
  the syntactic-vs-semantic gap claim to work.

- "Earth" as the suppressed word in the README pilot example:
  Removed. Confabulation on Claude Opus's part. The canonical example
  is planet→ocean with GALAXY as the secret-adjacent word, sourced from
  cs5374project log blind_log_google_genai_gemini-3.1-pro-preview_20260307_185702.txt
  and documented as Appendix C of the V&V final report.

## What's deferred (in rough priority order)

1. Fix test_temperature_conventions assertion (one-line, 30 seconds).
2. Master-summary scoring heuristic in batch runners ("confirmed +
   0.2 * prog" — needs policy decision on whether/how to weight static
   findings; defer until you have a reason to care about the master
   score).
3. Console tables and per-trial print lines in batch runners — pure
   observability, not schema. Fix when next batch run reveals them
   out of sync.
4. full_experiment_batch.py's Exp 3 CSV schema asymmetry: Exp 3 doesn't
   include programmatic_findings at all (orphaned writer). april7_batch
   handles the same CSV correctly. Decide whether to fix Exp 3 or
   delete it.


## Next session's intended starting point

The experiment roadmap is in docs/EXPERIMENT_ROADMAP.md. The order
that makes sense:

1. Cross-tool SAST replication (Semgrep + SonarQube against existing
   April 7 generated_code strings). Cheapest experiment, no LLM calls,
   strongest evidence for the "models defeat SAST" claim if the pattern
   holds across multiple tools.
2. Vulnerability-density task specs. Requires task spec design work.
   Goal is to get static layers actually firing so the gap is measurable
   rather than mostly-zero.
3. Adversarial generator framing ("write code that passes a security
   audit" vs "write secure code"). Direct test of the trained-against-
   rubrics hypothesis.

Do (1) first. It's the clearest win and validates whether bandit's
sparse hit rate is bandit-specific or a SAST-wide pattern.
