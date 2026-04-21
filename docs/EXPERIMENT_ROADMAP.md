# Domain 2 Experiment Roadmap

Captured 2026-04-20. Direction set after Phase 3 backfill revealed that
the static analysis layers (regex detector + bandit) fire sparsely on
the April 7 batch — most generated code was clean under both layers,
while the LLM referee carried the actual semantic load.

## Framing

Domain 2's contribution is reframed from "multi-layer architecture for
adversarial code review" to "measuring the gap between syntactic and
semantic defense in AI-generated code." The hypothesis: models trained
against static analysis rubrics produce code that passes static layers
but contains semantic vulnerabilities only the LLM layer can detect.

Future experiments are designed to test and quantify this gap.

## Planned experiments

### 1. Vulnerability-density task specs

Current task specs (SQL endpoint, SSRF endpoint) produce mostly-clean
generated code. Run the same architecture against task specs designed
to elicit vulnerabilities:
- Performance-framed prompts ("implement this efficiently") with no
  security framing
- Legacy-codebase-style specs that mention compatibility with old APIs
- Prompts that explicitly de-emphasize security ("quick prototype",
  "internal tool only")

Goal: get the static layers to actually fire so the gap between layers
is measurable rather than mostly-zero.

### 2. Cross-tool SAST replication

Run Semgrep and SonarQube against the same generated_code strings from
the April 7 batch (and any future batches). No LLM calls. Pure SAST
tool comparison. If all three static tools (regex, bandit, semgrep,
sonarqube) miss the same vulnerabilities the LLM catches, "models
defeat static analysis" becomes a multi-tool finding rather than a
bandit-specific one.

### 3. Adversarial generator framing

Same task spec, two generator prompts:
- "Write secure code that does X"
- "Write code that does X and passes a security audit"

Hypothesis: explicit "passes audit" framing produces code that's harder
for static layers to catch but no harder for the LLM. Direct test of
the "trained against static rubrics" hypothesis.

## Status

All three experiments are scoped, not started. Order is flexible —
(2) is cheapest, (1) requires task spec design work, (3) requires
careful prompt engineering for the generator.
