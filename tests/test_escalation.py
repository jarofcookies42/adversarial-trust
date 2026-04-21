"""Tests for the verdict escalation logic in domains.code_security.graph.

Covers the pure-function helpers that sit outside build_code_security_graph:

  - _rank_quality:          severity-ordering utility
  - _referee_covers:        heuristic coverage check
  - _apply_deterministic_escalation: the main escalation pass

No LLM calls, no graph instantiation, no file I/O — log_round is mocked so
tests don't touch disk. Works on the raw dicts the graph node passes around.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from domains.code_security.graph import (
    _QUALITY_ORDER,
    _SOURCE_TO_SUMMARY_KEY,
    _apply_deterministic_escalation,
    _rank_quality,
    _referee_covers,
)


# Autouse: silence log_round across all tests in this module so they don't
# write to disk or print to console.
@pytest.fixture(autouse=True)
def _mute_log_round():
    with patch("domains.code_security.graph.log_round"):
        yield


# ── _rank_quality ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("quality,expected_rank", [
    ("secure",        0),
    ("mostly_secure", 1),
    ("concerning",    2),
    ("vulnerable",    3),
    ("critical",      4),
    ("unknown",       5),
])
def test_rank_quality_known_values(quality, expected_rank):
    assert _rank_quality(quality) == expected_rank


def test_rank_quality_unknown_value_returns_minus_one():
    assert _rank_quality("gibberish") == -1
    assert _rank_quality("") == -1
    # Case-sensitive — lookup is exact-match
    assert _rank_quality("Secure") == -1


def test_rank_quality_ordering_is_strict_ascending():
    """Severity ordering: secure < mostly_secure < concerning < vulnerable < critical."""
    real_qualities = ["secure", "mostly_secure", "concerning", "vulnerable", "critical"]
    ranks = [_rank_quality(q) for q in real_qualities]
    assert ranks == sorted(ranks)
    assert len(set(ranks)) == len(ranks)  # all distinct
    # Specific inequalities the escalation logic depends on:
    assert _rank_quality("secure") < _rank_quality("vulnerable")
    assert _rank_quality("concerning") < _rank_quality("vulnerable")
    assert _rank_quality("critical") > _rank_quality("vulnerable")


def test_quality_order_constant_matches_documented_shape():
    """If someone reorders _QUALITY_ORDER, escalation semantics break silently.
    This test will flag any drift."""
    assert _QUALITY_ORDER == [
        "secure", "mostly_secure", "concerning", "vulnerable", "critical", "unknown"
    ]


# ── _referee_covers ───────────────────────────────────────────────────────

def _final_issue(**fields):
    """Shorthand — build a referee final_issue dict with arbitrary fields."""
    return fields


def test_referee_covers_matches_category():
    finding = {"category": "injection", "rule_id": "INJ-001", "match": "cursor.execute(x)"}
    final_issues = [_final_issue(explanation="SQL injection vulnerability at line 12")]
    assert _referee_covers(finding, final_issues) is True


def test_referee_covers_matches_via_dedicated_category_field():
    finding = {"category": "crypto", "rule_id": "", "match": ""}
    final_issues = [_final_issue(category="crypto", final_severity="high")]
    assert _referee_covers(finding, final_issues) is True


def test_referee_covers_matches_rule_id():
    finding = {"category": "", "rule_id": "B608", "match": ""}
    final_issues = [_final_issue(explanation="Bandit rule B608 fired")]
    assert _referee_covers(finding, final_issues) is True


def test_referee_covers_rule_id_match_is_case_insensitive():
    finding = {"category": "", "rule_id": "INJ-001", "match": ""}
    final_issues = [_final_issue(explanation="the INJ-001 rule was triggered")]
    assert _referee_covers(finding, final_issues) is True


def test_referee_covers_matches_substring_when_long_enough():
    finding = {"category": "", "rule_id": "", "match": "cursor.execute(user_input_query)"}
    final_issues = [_final_issue(
        explanation="The code calls cursor.execute(user_input_query) without parameterization"
    )]
    assert _referee_covers(finding, final_issues) is True


def test_referee_covers_substring_length_guard_rejects_short_match():
    """needle_match must be >6 characters to be considered — guards against
    trivially-common short strings matching everything."""
    finding = {"category": "", "rule_id": "", "match": "abc"}  # 3 chars
    final_issues = [_final_issue(explanation="contains abc here")]
    assert _referee_covers(finding, final_issues) is False

    finding6 = {"category": "", "rule_id": "", "match": "abcdef"}  # 6 chars — still rejected
    assert _referee_covers(finding6, final_issues) is False

    # 7 chars — accepted (> 6)
    finding7 = {"category": "", "rule_id": "", "match": "abcdefg"}
    assert _referee_covers(finding7, [_final_issue(explanation="foo abcdefg bar")]) is True


def test_referee_covers_no_match_returns_false():
    finding = {"category": "crypto", "rule_id": "CRYPTO-001", "match": "md5("}
    final_issues = [_final_issue(explanation="SQL injection at line 5", category="injection")]
    assert _referee_covers(finding, final_issues) is False


def test_referee_covers_empty_final_issues_returns_false():
    finding = {"category": "injection", "rule_id": "INJ-001", "match": "eval(user_in)"}
    assert _referee_covers(finding, []) is False


def test_referee_covers_skips_non_dict_entries():
    """Defensive: final_issues could contain strings or None from a malformed
    referee response. Must not raise."""
    finding = {"category": "injection", "rule_id": "INJ-001", "match": ""}
    final_issues = ["not a dict", None, 42, _final_issue(explanation="injection detected")]
    assert _referee_covers(finding, final_issues) is True  # still finds the real one


def test_referee_covers_handles_missing_needle_fields():
    """A finding with no category/rule_id/match fields should never match anything."""
    finding = {"category": "", "rule_id": "", "match": ""}
    final_issues = [_final_issue(explanation="anything here", category="anything")]
    assert _referee_covers(finding, final_issues) is False


# ── _apply_deterministic_escalation ───────────────────────────────────────

def _crit(rule_id="B608", category="injection", line=42, match="cursor.execute(user_q)"):
    return {
        "rule_id": rule_id,
        "severity": "critical",
        "category": category,
        "description": f"Critical {category} issue",
        "line": line,
        "match": match,
    }


def _medium(**kwargs):
    kwargs.setdefault("rule_id", "RES-001")
    kwargs.setdefault("category", "resource")
    return {**_crit(**kwargs), "severity": "medium"}


def _verdict(quality="concerning", final_issues=None, summary=None):
    return {
        "overall_code_quality": quality,
        "final_issues": list(final_issues) if final_issues else [],
        "summary": dict(summary) if summary else {},
    }


def test_empty_layers_no_mutation():
    verdict = _verdict(quality="concerning")
    before = dict(verdict)
    before["final_issues"] = list(verdict["final_issues"])
    before["summary"] = dict(verdict["summary"])

    _apply_deterministic_escalation(verdict, [], log_path="anything")

    assert verdict["overall_code_quality"] == "concerning"
    assert verdict["final_issues"] == []
    # No summary keys added
    assert "detector_escalations" not in verdict["summary"]
    assert "static_escalations" not in verdict["summary"]


def test_non_critical_findings_do_not_escalate():
    verdict = _verdict(quality="concerning")
    _apply_deterministic_escalation(
        verdict,
        [
            ("deterministic_detector", [_medium()]),
            ("static_scanner",         [_medium(rule_id="B104", category="config")]),
        ],
        log_path="anything",
    )
    assert verdict["overall_code_quality"] == "concerning"
    assert verdict["final_issues"] == []
    assert "detector_escalations" not in verdict["summary"]
    assert "static_escalations" not in verdict["summary"]


def test_critical_finding_already_covered_by_referee_no_escalation():
    """Referee's final_issues already mentions injection — _referee_covers
    short-circuits on category match and no escalation fires."""
    existing_issue = {
        "issue_id": "ref-1",
        "final_verdict": "confirmed",
        "final_severity": "high",
        "explanation": "Injection vulnerability confirmed at line 42",
    }
    verdict = _verdict(quality="vulnerable", final_issues=[existing_issue])
    _apply_deterministic_escalation(
        verdict,
        [("deterministic_detector", [_crit(category="injection")])],
        log_path="anything",
    )
    # No append happened — still just the one referee-provided issue
    assert len(verdict["final_issues"]) == 1
    assert verdict["final_issues"][0]["issue_id"] == "ref-1"
    assert verdict["overall_code_quality"] == "vulnerable"  # unchanged
    assert "detector_escalations" not in verdict["summary"]


def test_critical_missed_is_appended_with_correct_shape():
    verdict = _verdict(quality="concerning")
    finding = _crit(rule_id="B608", category="injection", line=42)
    _apply_deterministic_escalation(
        verdict,
        [("static_scanner", [finding])],
        log_path="anything",
    )

    assert len(verdict["final_issues"]) == 1
    added = verdict["final_issues"][0]
    assert added["issue_id"] == "static_scanner:B608:L42"
    assert added["final_verdict"] == "confirmed"
    assert added["final_severity"] == "critical"
    assert added["source"] == "static_scanner"
    assert "Added by static_scanner" in added["explanation"]
    assert "not confirmed by referee" in added["explanation"]


def test_critical_missed_lifts_quality_from_secure_to_vulnerable():
    verdict = _verdict(quality="secure")
    _apply_deterministic_escalation(
        verdict,
        [("deterministic_detector", [_crit(category="crypto")])],
        log_path="anything",
    )
    assert verdict["overall_code_quality"] == "vulnerable"


@pytest.mark.parametrize("start_quality", ["secure", "mostly_secure", "concerning"])
def test_quality_lifted_to_vulnerable_from_any_lower_tier(start_quality):
    verdict = _verdict(quality=start_quality)
    _apply_deterministic_escalation(
        verdict,
        [("deterministic_detector", [_crit(category="crypto")])],
        log_path="anything",
    )
    assert verdict["overall_code_quality"] == "vulnerable"


def test_quality_at_critical_is_not_downgraded():
    """Even when we escalate, we must never lower an existing severity."""
    verdict = _verdict(quality="critical")
    _apply_deterministic_escalation(
        verdict,
        [("static_scanner", [_crit(category="crypto")])],
        log_path="anything",
    )
    # Quality stays at critical — we only lift, never lower
    assert verdict["overall_code_quality"] == "critical"
    # But the escalation still ran — issue was appended + count recorded
    assert len(verdict["final_issues"]) == 1
    assert verdict["summary"]["static_escalations"] == 1


def test_quality_at_vulnerable_stays_vulnerable_with_missed_finding():
    verdict = _verdict(quality="vulnerable")
    _apply_deterministic_escalation(
        verdict,
        [("deterministic_detector", [_crit(category="crypto")])],
        log_path="anything",
    )
    assert verdict["overall_code_quality"] == "vulnerable"
    assert verdict["summary"]["detector_escalations"] == 1


def test_multiple_layers_with_mixed_coverage_counts_correctly():
    """
    Setup:
      - Referee already mentions 'injection' — so any injection-category
        finding is considered covered.
      - Detector contributes 2 critical findings: 1 injection (covered),
        1 crypto (missed).
      - Static scanner contributes 2 critical findings: both crypto (both
        missed, since 'injection' match doesn't cover 'crypto').

    Expected:
      - detector_escalations == 1
      - static_escalations   == 2
      - total appended: 3
      - quality lifted to vulnerable (was concerning)
    """
    existing = [{
        "issue_id": "ref-1",
        "final_verdict": "confirmed",
        "final_severity": "high",
        "explanation": "SQL injection detected at line 5",
    }]
    verdict = _verdict(quality="concerning", final_issues=existing)

    _apply_deterministic_escalation(
        verdict,
        [
            ("deterministic_detector", [
                _crit(rule_id="INJ-001", category="injection", line=5,
                      match="cursor.execute(q)"),   # covered by referee
                _crit(rule_id="CRYPTO-002", category="crypto", line=20,
                      match="DES(key)"),            # missed
            ]),
            ("static_scanner", [
                _crit(rule_id="B324", category="crypto", line=10,
                      match="hashlib.md5(x)"),      # missed
                _crit(rule_id="B505", category="crypto", line=15,
                      match="weak_key_size"),       # missed
            ]),
        ],
        log_path="anything",
    )

    assert verdict["summary"]["detector_escalations"] == 1
    assert verdict["summary"]["static_escalations"] == 2
    # 1 original referee issue + 3 escalated = 4 total
    assert len(verdict["final_issues"]) == 4

    appended = [i for i in verdict["final_issues"] if i.get("source")]
    sources = [i["source"] for i in appended]
    assert sources.count("deterministic_detector") == 1
    assert sources.count("static_scanner") == 2

    assert verdict["overall_code_quality"] == "vulnerable"


def test_source_tag_in_map_used_for_summary_key():
    """Summary uses _SOURCE_TO_SUMMARY_KEY to translate source -> key."""
    # Sanity: the map has the two expected layers
    assert _SOURCE_TO_SUMMARY_KEY["deterministic_detector"] == "detector_escalations"
    assert _SOURCE_TO_SUMMARY_KEY["static_scanner"] == "static_escalations"


def test_unknown_source_tag_falls_back_to_autogenerated_summary_key():
    """Future layers (semgrep, CodeQL) may pass a source tag not in the map.
    The helper should still record a count — it derives a key from the tag."""
    verdict = _verdict(quality="concerning")
    _apply_deterministic_escalation(
        verdict,
        [("hypothetical_semgrep", [_crit(category="crypto")])],
        log_path="anything",
    )
    # Auto-derived key: "<tag>_escalations"
    assert verdict["summary"]["hypothetical_semgrep_escalations"] == 1
    assert verdict["overall_code_quality"] == "vulnerable"


def test_issue_id_format_preserves_rule_id_and_line():
    """issue_id shape: '{source_tag}:{rule_id}:L{line}' — this is the
    provenance signal analysis scripts will grep for."""
    verdict = _verdict(quality="secure")
    _apply_deterministic_escalation(
        verdict,
        [("static_scanner", [_crit(rule_id="B608", line=123)])],
        log_path="anything",
    )
    assert verdict["final_issues"][0]["issue_id"] == "static_scanner:B608:L123"


def test_issue_id_handles_missing_rule_id_and_line_gracefully():
    verdict = _verdict(quality="secure")
    bare_finding = {
        "severity": "critical",
        "category": "injection",
        "description": "minimal finding",
        # rule_id, line, match all missing
    }
    _apply_deterministic_escalation(
        verdict,
        [("static_scanner", [bare_finding])],
        log_path="anything",
    )
    # Helper substitutes '?' for both missing pieces
    assert verdict["final_issues"][0]["issue_id"] == "static_scanner:?:L?"
