"""LangGraph orchestration for the code security adversarial review.

Flow: generate_code → find_issues → challenge_issues → referee_verdict → save_results

This is the core graph for Chapter 2 of the thesis.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from maat.config import TEMP_ATTACKER, TEMP_JUDGE, TEMP_TARGET, ModelConfig, RECURSION_LIMIT
from maat.models import get_model
from maat.state import CodeSecurityState
from maat.utils import console, log_round, make_log_path, make_result_path, save_result

from domains.code_security.detector import run_detector, severity_score
from domains.code_security.static_scanner import run_static_scanner
from domains.code_security.prompts import (
    ADVERSARY_SYSTEM,
    FINDER_SYSTEM,
    GENERATOR_SYSTEM,
    REFEREE_SYSTEM,
)


# ── Verdict escalation helper ──────────────────────────────────────────────
#
# Deterministic layers (regex detector, bandit static scanner) can flag
# critical issues with certainty. When the probabilistic LLM referee misses
# one, we escalate: tag the finding as confirmed, raise overall quality to
# at least "vulnerable", record per-layer counts in the summary.
#
# This helper handles N layers uniformly so adding a future scanner
# (semgrep, CodeQL, etc.) is additive — no new branches needed.

_QUALITY_ORDER = ["secure", "mostly_secure", "concerning",
                  "vulnerable", "critical", "unknown"]


def _rank_quality(q: str) -> int:
    try:
        return _QUALITY_ORDER.index(q)
    except ValueError:
        return -1


def _referee_covers(finding: dict, final_issues: list[dict]) -> bool:
    """Heuristic: does the referee's final_issues list already acknowledge
    this deterministic finding? Checks category, rule_id, and match text
    against the referee's natural-language explanations."""
    needle_cat = (finding.get("category") or "").lower()
    needle_match = (finding.get("match") or "").lower()
    needle_rule = (finding.get("rule_id") or "").lower()
    for fi in final_issues:
        if not isinstance(fi, dict):
            continue
        hay = " ".join(
            str(fi.get(k, "")) for k in
            ("explanation", "description", "type", "category", "final_severity")
        ).lower()
        if needle_cat and needle_cat in hay:
            return True
        if needle_rule and needle_rule in hay:
            return True
        if needle_match and len(needle_match) > 6 and needle_match in hay:
            return True
    return False


# Per-layer source tags, used both as the `source` field on appended issues
# and as the key in the summary escalation counts. Kept in one place so the
# strings don't drift.
_SOURCE_TO_SUMMARY_KEY = {
    "deterministic_detector": "detector_escalations",
    "static_scanner":          "static_escalations",
}


def _apply_deterministic_escalation(
    verdict: dict,
    layers: list[tuple[str, list[dict]]],
    log_path,
) -> None:
    """Append critical findings the referee missed to verdict['final_issues'],
    escalate overall_code_quality if any were added, record per-layer counts.

    Mutates `verdict` in place. `layers` is a list of (source_tag, findings)
    tuples where source_tag is a key in _SOURCE_TO_SUMMARY_KEY.
    """
    final_issues = verdict.setdefault("final_issues", [])
    per_layer_missed: dict[str, list[dict]] = {}
    total = 0

    for source_tag, findings in layers:
        missed = [
            f for f in findings
            if f.get("severity") == "critical" and not _referee_covers(f, final_issues)
        ]
        per_layer_missed[source_tag] = missed
        total += len(missed)

    if total == 0:
        return

    for source_tag, missed in per_layer_missed.items():
        for f in missed:
            final_issues.append({
                "issue_id": f"{source_tag}:{f.get('rule_id', '?')}:L{f.get('line', '?')}",
                "final_verdict": "confirmed",
                "final_severity": f.get("severity", "critical"),
                "explanation": (
                    f"Added by {source_tag} (not confirmed by referee): "
                    f"{f.get('description', '')}. "
                    f"Match: {(f.get('match') or '')[:120]}"
                ),
                "source": source_tag,
            })

    current_q = verdict.get("overall_code_quality", "unknown")
    if _rank_quality(current_q) < _rank_quality("vulnerable"):
        verdict["overall_code_quality"] = "vulnerable"

    summary = verdict.setdefault("summary", {})
    for source_tag, missed in per_layer_missed.items():
        key = _SOURCE_TO_SUMMARY_KEY.get(source_tag, f"{source_tag}_escalations")
        summary[key] = len(missed)

    try:
        breakdown = ", ".join(
            f"{tag}={len(m)}" for tag, m in per_layer_missed.items() if m
        )
        log_round(log_path, "escalation",
                  f"Deterministic layers escalated {total} critical finding(s) "
                  f"missed by referee ({breakdown}); "
                  f"quality -> {verdict['overall_code_quality']}")
    except Exception:
        pass

# Retry config for rate-limited APIs
MAX_RETRIES = 3
BASE_BACKOFF = 30  # seconds


def _invoke_with_retry(llm: Any, messages: list, role: str) -> Any:
    """Invoke an LLM with exponential backoff on rate-limit errors."""
    attempts = MAX_RETRIES + 1  # total attempts including the final one
    for attempt in range(attempts):
        try:
            return llm.invoke(messages)
        except Exception as e:
            err = str(e).lower()
            is_rate_limit = "429" in err or "rate" in err or "quota" in err or "resource_exhausted" in err
            is_timeout = "timeout" in err or "timed out" in err or "deadline" in err
            if (is_rate_limit or is_timeout) and attempt < attempts - 1:
                wait = BASE_BACKOFF * (2 ** attempt)
                reason = "Rate limited" if is_rate_limit else "Timeout"
                console.print(f"[yellow]{role}: {reason}, retrying in {wait}s (attempt {attempt + 1}/{attempts})[/yellow]")
                time.sleep(wait)
            else:
                raise


def _get_text(response: Any) -> str:
    """Extract plain text from an LLM response.

    Gemini 3.x models return response.content as a list of content blocks
    like [{"type": "text", "text": "..."}] instead of a plain string.
    This normalizes both formats.
    """
    content = response.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content)


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences from generated code."""
    stripped = text.strip()
    # Remove opening fence with optional language tag
    stripped = re.sub(r'^```\w*\n?', '', stripped)
    # Remove closing fence
    stripped = re.sub(r'\n?```$', '', stripped)
    return stripped.strip()


def _parse_json_from_response(text: str, fallback_key: str = "data") -> Any:
    """Extract JSON from an LLM response that may include markdown fences.

    Handles: plain JSON, fenced JSON blocks, JSON embedded in prose,
    and multiple fenced blocks (picks the first valid JSON).
    """
    # Strategy 1: Try to find JSON inside ```json ... ``` fences
    fenced = re.findall(r'```(?:json)?\s*\n?([\s\S]*?)\n?```', text)
    for block in fenced:
        block = block.strip()
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            continue

    # Strategy 2: Try the full text stripped of fences
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r'^```\w*\n?', '', cleaned)
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Strategy 3: Find the largest JSON array or object in the text
    best = None
    best_len = 0
    for start_char, end_char in [("[", "]"), ("{", "}")]:
        start = 0
        while True:
            start = text.find(start_char, start)
            if start == -1:
                break
            # Find the matching closing character
            end = text.rfind(end_char)
            if end != -1 and end > start:
                candidate = text[start:end + 1]
                try:
                    parsed = json.loads(candidate)
                    if len(candidate) > best_len:
                        best = parsed
                        best_len = len(candidate)
                except json.JSONDecodeError:
                    pass
            start += 1

    if best is not None:
        return best

    # Give up — return raw text wrapped
    return {fallback_key: text}


def _extract_list(parsed: Any, keys: tuple[str, ...] = ()) -> list:
    """Unwrap a parsed JSON response into a list.

    Handles: plain list, dict with a known key containing a list,
    or dict with any single list-valued key.
    """
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        # Check known keys first
        for k in keys:
            if k in parsed and isinstance(parsed[k], list):
                return parsed[k]
        # Fall back to any key that holds a list
        for v in parsed.values():
            if isinstance(v, list):
                return v
    return []


def build_code_security_graph(
    generator_config: ModelConfig,
    finder_config: ModelConfig | None = None,
    adversary_config: ModelConfig | None = None,
    referee_config: ModelConfig | None = None,
) -> StateGraph:
    """Build the code security adversarial review graph.

    If finder/adversary/referee configs aren't specified, they default
    to the generator model with appropriate temperatures.
    """
    # Default: use same model for all roles with different temperatures
    if finder_config is None:
        finder_config = ModelConfig(
            provider=generator_config.provider,
            model_name=generator_config.model_name,
            temperature=TEMP_ATTACKER,
        )
    if adversary_config is None:
        adversary_config = ModelConfig(
            provider=generator_config.provider,
            model_name=generator_config.model_name,
            temperature=TEMP_ATTACKER,
        )
    if referee_config is None:
        referee_config = ModelConfig(
            provider=generator_config.provider,
            model_name=generator_config.model_name,
            temperature=TEMP_JUDGE,
        )

    generator_llm = get_model(generator_config)
    finder_llm = get_model(finder_config)
    adversary_llm = get_model(adversary_config)
    referee_llm = get_model(referee_config)

    log_path = make_log_path("code_security", generator_config.display_name)

    def generate_code(state: CodeSecurityState) -> dict:
        """Generator produces code from the task spec."""
        messages = [
            SystemMessage(content=GENERATOR_SYSTEM),
            HumanMessage(content=f"Task: {state['task_spec']}"),
        ]
        response = _invoke_with_retry(generator_llm, messages, "generator")
        code = _strip_code_fences(_get_text(response))
        log_round(log_path, "generator", code)
        return {
            "generated_code": code,
            "conversation_history": [{"role": "generator", "content": code}],
        }

    def find_issues(state: CodeSecurityState) -> dict:
        """Finder reviews the code and reports all issues."""
        messages = [
            SystemMessage(content=FINDER_SYSTEM),
            HumanMessage(content=f"Review this code for security issues:\n\n```\n{state['generated_code']}\n```"),
        ]
        response = _invoke_with_retry(finder_llm, messages, "finder")
        raw = _get_text(response)
        log_round(log_path, "finder", raw)

        parsed = _parse_json_from_response(raw, fallback_key="issues")
        issues = _extract_list(parsed, ("issues",))

        return {
            "finder_issues": issues,
            "conversation_history": [{"role": "finder", "content": raw}],
        }

    def challenge_issues(state: CodeSecurityState) -> dict:
        """Adversary challenges each finding."""
        issues = state.get("finder_issues", [])
        if not issues:
            return {
                "adversary_challenges": [],
                "surviving_issues": [],
                "conversation_history": [{"role": "adversary", "content": "No issues to challenge."}],
            }

        messages = [
            SystemMessage(content=ADVERSARY_SYSTEM),
            HumanMessage(content=(
                f"The code:\n```\n{state['generated_code']}\n```\n\n"
                f"Issues found:\n{json.dumps(issues, indent=2)}\n\n"
                f"Challenge each issue. Kill the weak ones."
            )),
        ]
        response = _invoke_with_retry(adversary_llm, messages, "adversary")
        raw = _get_text(response)
        log_round(log_path, "adversary", raw)

        parsed = _parse_json_from_response(raw, fallback_key="challenges")
        challenges = _extract_list(parsed, ("challenges",))

        # Determine surviving issues
        killed_ids = set()
        for c in challenges:
            if isinstance(c, dict) and c.get("verdict") == "killed":
                killed_ids.add(c.get("issue_id"))

        surviving = [i for i in issues if i.get("id") not in killed_ids]

        return {
            "adversary_challenges": challenges,
            "surviving_issues": surviving,
            "conversation_history": [{"role": "adversary", "content": raw}],
        }

    def referee_verdict(state: CodeSecurityState) -> dict:
        """Referee delivers final ground truth."""
        messages = [
            SystemMessage(content=REFEREE_SYSTEM),
            HumanMessage(content=(
                f"Original code:\n```\n{state['generated_code']}\n```\n\n"
                f"Finder's issues:\n{json.dumps(state.get('finder_issues', []), indent=2)}\n\n"
                f"Adversary's challenges:\n{json.dumps(state.get('adversary_challenges', []), indent=2)}\n\n"
                f"Deliver your final verdict."
            )),
        ]
        response = _invoke_with_retry(referee_llm, messages, "referee")
        raw = _get_text(response)
        log_round(log_path, "referee", raw)

        verdict = _parse_json_from_response(raw, fallback_key="verdict")
        if not isinstance(verdict, dict) or "final_issues" not in verdict:
            verdict = {"raw": raw, "final_issues": [], "summary": {}, "overall_code_quality": "unknown"}

        # Fallback: if referee returned empty final_issues but surviving_issues exist,
        # use surviving issues as the verdict. This handles referee truncation on large inputs.
        surviving = state.get("surviving_issues", [])
        if not verdict.get("final_issues") and surviving:
            console.print("[yellow]WARNING: Referee returned empty final_issues — "
                          f"falling back to {len(surviving)} surviving issues from adversary stage[/yellow]")
            log_round(log_path, "warning",
                      f"Referee truncation fallback: using {len(surviving)} surviving issues")
            verdict["final_issues"] = [
                {
                    "issue_id": issue.get("id"),
                    "final_verdict": "confirmed",
                    "final_severity": issue.get("severity", "medium"),
                    "explanation": f"Fallback from adversary stage: {issue.get('description', 'N/A')}",
                }
                for issue in surviving
            ]
            verdict["summary"] = {
                "total_found": len(state.get("finder_issues", [])),
                "confirmed": len(surviving),
                "rejected": len(state.get("finder_issues", [])) - len(surviving),
                "modified": 0,
                "finder_accuracy": round(len(surviving) / max(1, len(state.get("finder_issues", []))), 2),
                "adversary_accuracy": 0.0,  # unknown — referee failed
                "referee_fallback": True,
            }
            verdict["overall_code_quality"] = (
                "critical" if any(i.get("severity") == "critical" for i in surviving)
                else "vulnerable" if any(i.get("severity") == "high" for i in surviving)
                else "concerning"
            )

        # ── Deterministic layers ──────────────────────────────────────────
        # Two independent scanners. Kept as separate fields in the result JSON
        # so per-layer analysis stays possible — do NOT merge these lists.
        #   - programmatic_findings: regex detector (line-local textual patterns)
        #   - static_findings:        bandit (AST-level structural patterns)

        prog_findings = run_detector(state["generated_code"])
        prog_score = severity_score(prog_findings)
        if prog_findings:
            log_round(log_path, "detector",
                      f"Regex scan: {len(prog_findings)} findings, severity score {prog_score}")

        static_findings = run_static_scanner(state["generated_code"])
        static_score = severity_score(static_findings)
        if static_findings:
            log_round(log_path, "static",
                      f"Static scan: {len(static_findings)} findings, severity score {static_score}")

        # Generalized escalation across all deterministic layers — critical
        # findings the referee missed get appended to final_issues and lift
        # overall_code_quality to at least "vulnerable".
        _apply_deterministic_escalation(
            verdict,
            [
                ("deterministic_detector", prog_findings),
                ("static_scanner",          static_findings),
            ],
            log_path,
        )

        # Save full result
        result = {
            "domain": "code_security",
            "task_spec": state["task_spec"],
            "generator": generator_config.display_name,
            "finder": finder_config.display_name,
            "adversary": adversary_config.display_name,
            "referee": referee_config.display_name,
            "generated_code": state["generated_code"],
            "finder_issues": state.get("finder_issues", []),
            "adversary_challenges": state.get("adversary_challenges", []),
            "surviving_issues": state.get("surviving_issues", []),
            "referee_verdict": verdict,
            "programmatic_findings": prog_findings,
            "programmatic_severity_score": prog_score,
            "static_findings": static_findings,
            "static_severity_score": static_score,
        }
        result_path = make_result_path("code_security", generator_config.display_name)
        save_result(result_path, result)

        return {
            "referee_verdict": verdict,
            "programmatic_findings": prog_findings,
            "static_findings": static_findings,
            "conversation_history": [{"role": "referee", "content": raw}],
        }

    # Build the graph
    graph = StateGraph(CodeSecurityState)
    graph.add_node("generate_code", generate_code)
    graph.add_node("find_issues", find_issues)
    graph.add_node("challenge_issues", challenge_issues)
    graph.add_node("referee_verdict", referee_verdict)

    graph.set_entry_point("generate_code")
    graph.add_edge("generate_code", "find_issues")
    graph.add_edge("find_issues", "challenge_issues")
    graph.add_edge("challenge_issues", "referee_verdict")
    graph.add_edge("referee_verdict", END)

    return graph.compile()
