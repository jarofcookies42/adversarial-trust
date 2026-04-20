# MAAT — Multi-Agent Adversarial Trustworthiness

A generalized framework for evaluating AI system trustworthiness through competing AI agents.

## What is this?

MAAT pits AI agents against each other in structured adversarial engagements to test whether AI systems can be trusted. The core insight: **you can learn more about a system's security by watching it defend than by reading its documentation.**

The framework applies the same multi-agent pattern across multiple trust dimensions:

| Domain | Generator | Attacker | Judge | Question Answered |
|--------|-----------|----------|-------|-------------------|
| **Secret Extraction** | — | Social engineering agent | Leak detection (LLM + programmatic) | Can the model keep a secret under adversarial pressure? |
| **Code Security** | Code writing agent | Vulnerability finder | Adversary + Referee | Does AI-generated code contain exploitable vulnerabilities? |
| **Persuasion Resistance** | — | Cialdini-principle attacker | Compliance evaluator | Can the model resist psychologically-grounded manipulation? |

## Architecture

Every domain follows the same competing-incentive pattern:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  GENERATOR  │────▶│   FINDER    │────▶│  ADVERSARY  │────▶│   REFEREE   │
│             │     │ +1 per issue│     │ +1 per kill │     │ +1 accuracy │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                          │                    │                    │
                     finds issues        disproves fakes      final verdict
                     (even borderline)   (attacks weak ones)  (ground truth)
```

**Why competing incentives?** When you ask one AI to review code, it finds bugs — even if it has to make some up (sycophancy). When you add an adversary that gets rewarded for disproving those bugs, the fake ones get killed. Three sycophantic agents pointed in different directions cancel each other out.

## Key Finding from Pilot Study

The pilot study (~90 trials across 10 models) discovered that **LLM defense mechanisms create observable side-channels**:

- Models actively suppress words similar to protected secrets in their output
- Chain-of-thought reasoning reveals vocabulary avoidance patterns (e.g., replacing "crunch" with "rustle" because "crush" is in the secret)
- **Active suppression is more visible than passive leakage** — the act of hiding IS the tell

This finding — that the defense process itself creates detectable signals — has not been documented in prior jailbreak literature and is the primary novel contribution.

## Quick Start

```bash
# Install uv first: `brew install uv` (or see https://astral.sh/uv)

# Clone and install
git clone <repo-url>
cd adversarial-trust
uv sync

# Set up environment
cp .env.example .env
# Edit .env with your API keys

# Run a code security review
uv run python -m runners.single_run \
  --domain code_security \
  --generator gemini-2.5-flash \
  --task "implement a JWT authentication system in Python"

# Run a secret extraction test
uv run python -m runners.single_run \
  --domain secret_extraction \
  --target gemini-2.5-flash \
  --rounds 10
```

## Supported Models

| Provider | Models | Config |
|----------|--------|--------|
| **Ollama** (local) | llama3.2:3b, mistral, nous-hermes2 | `OLLAMA_BASE_URL` |
| **Google Gemini** | gemini-2.5-flash, gemini-3.1-pro, etc. | `GOOGLE_API_KEY` |
| **Anthropic** | claude-haiku-4.5, claude-sonnet-4.6 | `ANTHROPIC_API_KEY` |
| **OpenAI** | gpt-4o, o4-mini | `OPENAI_API_KEY` |

## Project Structure

```
adversarial-trust/
├── maat/                      # Core framework (model abstraction, state, graph builder)
├── domains/
│   ├── secret_extraction/     # LLM secret-keeping tests
│   ├── code_security/         # Adversarial code review
│   └── persuasion/            # Persuasion resistance (planned)
├── runners/                   # Test execution (single, batch, comparison)
├── analysis/                  # Metrics, side-channel analysis, plots
├── tests/                     # pytest suite
├── logs/                      # Test output (gitignored)
└── results/                   # Structured JSON/CSV results (gitignored)
```

## Research Context

This framework supports a master's thesis at Texas Tech University under Dr. Akbar Siami Namin, extending published work on LLM exploitation through deception techniques (Singh, Abri & Namin, IEEE BigData 2023; Singh & Namin, Computers in Human Behavior, 2025).

## License

MIT
