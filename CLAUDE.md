# CLAUDE.md — Instructions for Claude Code

## Project Overview

This is **MAAT** (Multi-Agent Adversarial Trustworthiness) — a generalized framework for evaluating AI system trustworthiness through competing AI agents. It extends a completed pilot study on LLM secret extraction (~90 trials, 10 models) into a multi-domain adversarial testing platform.

The architecture uses LangGraph for multi-agent orchestration and supports multiple LLM backends (Ollama local models, Google Gemini, Anthropic Claude, OpenAI).

## Architecture

The core pattern is always the same across all test domains:

```
[GENERATOR] → produces an artifact (code, text, response)
[ATTACKER/FINDER] → tries to find flaws, extract secrets, or exploit weaknesses
[DEFENDER] → (optional) tries to harden the artifact or detect attacks
[JUDGE/REFEREE] → evaluates what happened, scores severity
```

Each domain (secret-keeping, code security, persuasion resistance) implements this pattern with domain-specific prompts, evaluation criteria, and detection logic.

## Project Structure

```
adversarial-trust/
├── CLAUDE.md                  # This file
├── README.md                  # Project overview and usage
├── pyproject.toml             # Python project config (use uv or pip)
├── requirements.txt           # Dependencies
│
├── maat/                      # Core framework
│   ├── __init__.py
│   ├── config.py              # Model configs, API keys, settings
│   ├── models.py              # LLM provider abstraction (Ollama, Gemini, Claude, OpenAI)
│   ├── state.py               # Shared state definitions (TypedDict for LangGraph)
│   ├── graph.py               # Base graph builder — shared orchestration logic
│   ├── judge.py               # Base judge with dual-layer eval (LLM + programmatic)
│   └── utils.py               # Logging, file I/O, result formatting
│
├── domains/                   # Domain-specific implementations
│   ├── __init__.py
│   │
│   ├── secret_extraction/     # Chapter 1: LLM secret-keeping (ported from pilot)
│   │   ├── __init__.py
│   │   ├── prompts.py         # Target system prompts (v1 through v7_hardened)
│   │   ├── attacker.py        # Attack strategies (social engineering, 20Q, roleplay)
│   │   ├── judge.py           # Secret-specific judge (word matching + LLM eval)
│   │   ├── detector.py        # Programmatic leak detection (word match, confirmation, category)
│   │   └── graph.py           # Domain graph: attacker → target → judge loop
│   │
│   ├── code_security/         # Chapter 2: Adversarial code review
│   │   ├── __init__.py
│   │   ├── prompts.py         # Generator prompts, attacker prompts, referee prompts
│   │   ├── generator.py       # Code generation agent (writes code with task spec)
│   │   ├── finder.py          # Vulnerability finder agent (+1 per issue found)
│   │   ├── adversary.py       # Adversary agent (tries to disprove findings, +1 per kill)
│   │   ├── referee.py         # Final verdict agent (+1 for accuracy)
│   │   ├── detector.py        # Programmatic checks (static analysis, pattern matching)
│   │   └── graph.py           # Domain graph: generate → find → challenge → referee
│   │
│   └── persuasion/            # Chapter 3: Persuasion resistance (future)
│       ├── __init__.py
│       └── ...
│
├── runners/                   # Test execution
│   ├── __init__.py
│   ├── single_run.py          # Run one test with specified domain + config
│   ├── batch_runner.py        # Run N trials across configs, save results
│   └── compare.py             # Compare results across models/prompts/domains
│
├── analysis/                  # Result analysis and visualization
│   ├── __init__.py
│   ├── metrics.py             # Breach rates, danger scores, information leakage
│   ├── side_channels.py       # Chain-of-thought analysis for suppression artifacts
│   └── plots.py               # Visualization helpers
│
├── tests/                     # pytest tests
│   ├── test_judge.py
│   ├── test_detector.py
│   ├── test_state.py
│   └── test_graph.py
│
├── logs/                      # Test output logs (gitignored)
└── results/                   # Structured results JSON/CSV (gitignored)
```

## Tech Stack

- **Python 3.12+**
- **LangGraph** for multi-agent orchestration
- **LangChain** for LLM provider abstraction
- **LangSmith** for tracing and evaluation (optional but recommended)
- **Ollama** for local model inference
- **google-genai** for Gemini models
- **anthropic** for Claude models
- **pytest** for testing
- **uv** preferred for package management (pip works too)

## Key Design Decisions

1. **Domain-agnostic core, domain-specific implementations.** The `maat/` package provides the shared orchestration, state management, and evaluation infrastructure. Each domain in `domains/` implements the pattern with its own prompts, agents, and scoring.

2. **Dual-layer evaluation everywhere.** Every domain must have both an LLM judge AND programmatic detection. The pilot study proved LLM judges miss obvious things (Day 1: full leak scored 6/10 PARTIAL by the judge). Programmatic detection catches what the LLM judge misses; the LLM judge catches subtle/semantic things programmatic checks can't.

3. **Competing incentives for code review.** The code_security domain uses the Adversarial Review pattern: Finder is rewarded for finding issues (+1 per issue), Adversary is rewarded for disproving issues (+1 per kill), Referee is rewarded for accuracy. Competing incentives cancel out sycophancy.

4. **All results are structured JSON.** Every test run produces a JSON file with full conversation history, judge evaluations, programmatic detection results, metadata (model, temperature, timestamp, domain), and a summary. This enables automated analysis.

5. **Temperature conventions.** Target/generator: 0.3 (deterministic). Attacker/finder: 0.9 (creative). Judge/referee: 0.1 (consistent evaluation).

## Coding Standards

- Type hints on all function signatures
- Docstrings on all public functions
- Use `TypedDict` for LangGraph state (not dataclasses)
- Use `Annotated[list, operator.add]` for accumulator fields in state
- Async where possible for API calls
- All prompts in dedicated `prompts.py` files, never inline
- Logging to both console and file simultaneously
- No hardcoded API keys — use environment variables or .env file

## Environment Variables

```
LANGCHAIN_TRACING_V2=true          # Enable LangSmith tracing
LANGCHAIN_API_KEY=lsv2_pt_...      # LangSmith API key
LANGCHAIN_PROJECT=maat              # LangSmith project name
GOOGLE_API_KEY=...                  # For Gemini models
ANTHROPIC_API_KEY=...               # For Claude models
OPENAI_API_KEY=ollama               # Set to "ollama" for local routing
OLLAMA_BASE_URL=http://localhost:11434  # Ollama endpoint
```

## Running

```bash
# Single test
uv run python -m runners.single_run --domain secret_extraction --target gemini-2.5-flash --rounds 10

# Single code review
uv run python -m runners.single_run --domain code_security --generator gemini-2.5-flash --task "implement a user auth system"

# Batch
uv run python -m runners.batch_runner --domain code_security --config configs/code_review_batch.yaml
```

## What To Build First

Priority order:
1. `maat/config.py` and `maat/models.py` — get LLM providers working
2. `maat/state.py` — shared state definitions
3. `domains/code_security/` — the new Chapter 2 domain (most exciting, least done)
4. Port `domains/secret_extraction/` from the existing pilot project
5. `runners/single_run.py` — basic test execution
6. Tests

## Important Context

This project is for a master's thesis at Texas Tech University under Dr. Akbar Siami Namin. It builds on a completed pilot study (CS 5374 class project) that tested LLM secret-keeping and found novel side-channel evidence in chain-of-thought reasoning. The existing pilot code lives in a separate repo (blakedmoos/cs5374project) and will be ported into `domains/secret_extraction/`.

The LangGraph recursion limit should be set to 200 (not the default 50) for longer attack/review sequences.
