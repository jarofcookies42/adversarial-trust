"""Shared state definitions for LangGraph agents."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict


class BaseTestState(TypedDict, total=False):
    """Base state shared across all domains."""
    domain: str
    conversation_history: Annotated[list[dict], operator.add]
    round_number: int
    max_rounds: int
    judge_evaluations: Annotated[list[dict], operator.add]
    result_summary: dict
    metadata: dict  # model names, temperatures, timestamps, etc.


class SecretExtractionState(BaseTestState, total=False):
    """State for secret extraction domain."""
    secret: str
    secret_words: list[str]
    attacker_message: str
    target_response: str
    leak_status: Literal["NO_LEAK", "PARTIAL_LEAK", "FULL_LEAK"]
    danger_score: int  # 0-10
    words_leaked: list[str]
    secret_extracted: bool


class CodeSecurityState(BaseTestState, total=False):
    """State for code security adversarial review domain."""
    task_spec: str                     # What the generator was asked to build
    generated_code: str                # Code produced by generator
    finder_issues: list[dict]          # Issues found by finder [{description, severity, line, type}]
    adversary_challenges: list[dict]   # Adversary's rebuttals [{issue_id, challenge, verdict}]
    surviving_issues: list[dict]       # Issues that survived adversary challenge
    referee_verdict: dict              # Final evaluation {real_issues, false_positives, score}
    programmatic_findings: list[dict]  # Static analysis / pattern matching results
