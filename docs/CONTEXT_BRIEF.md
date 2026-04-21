# Context Brief

You are picking up work on the **adversarial-trust** repo, which
contains Domain 2 (code security review) of the MAAT framework — a
master's thesis project on AI-on-AI evaluation at Texas Tech under
Dr. Akbar Siami Namin.

## Read first

Before responding, read **docs/PROJECT_STATE.md** in this repo. It
captures current state, framing decisions, what's been considered
and rejected, and what's deferred. The README and CLAUDE.md describe
what the code does; PROJECT_STATE captures why.

## Critical framing

Domain 2's thesis contribution is "measuring the gap between syntactic
and semantic defense in AI-generated code" — NOT "multi-layer
architecture for adversarial code review." Static analysis layers fire
sparsely because models have been trained against SAST rubrics; the
LLM referee carries the semantic load. That gap is the finding.

Don't reintroduce the "competing incentives / +1 per issue" framing
even if you see traces of it in older logs or commits — it's
prose-level role-conditioning, not engineered game theory. The current
README and CLAUDE.md reflect the corrected framing.

## Operational notes

- Python 3.13 via uv. Run commands as `uv run python -m runners.X`.
- Don't touch /Users/Jack/Desktop/projects/cs5374project (separate
  critical project with thesis data, treated as read-only).
- Don't touch results/ or logs/ without explicit instruction.
- Test suite: `uv run pytest`. 73 passing, 1 pre-existing failure.
- Stop and ask between phases of any multi-step work. Long prompts
  with hard guardrails are appropriate when operations are
  destructive; short prompts are fine for routine work.

## Sister repo

cs5374project (Domain 1 — secret extraction pilot) lives at
/Users/Jack/Desktop/projects/cs5374project. It contains the canonical
experimental data grounding most thesis claims. Read-only from
adversarial-trust's perspective.
