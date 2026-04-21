"""Tests for domains.code_security.static_scanner.

Covers:
  - Known-vulnerable fixtures fire the expected bandit rule_ids, with the
    expected severity after confidence-shifted mapping
  - Empty input returns []
  - Graceful failure when bandit is unavailable (no raise, returns [])
  - Every finding dict has the full 6-key Finding shape
"""

from __future__ import annotations

import subprocess
from unittest.mock import patch

import pytest

from domains.code_security.static_scanner import (
    _map_category,
    _map_severity,
    run_static_scanner,
    severity_score,
)


# ── Shared fixtures ───────────────────────────────────────────────────────

FLASK_DEBUG_BIND_ALL = """\
from flask import Flask
app = Flask(__name__)
app.run(debug=True, host="0.0.0.0", port=5000)
"""

FSTRING_SQL_CROSS_LINE = """\
import psycopg2
def find_user(conn, name):
    query = f"SELECT * FROM users WHERE name = '{name}'"
    with conn.cursor() as cur:
        cur.execute(query)
        return cur.fetchall()
"""

MD5_PICKLE_EVAL = """\
import hashlib, pickle
def h(u):
    return hashlib.md5(u.encode()).hexdigest()
def load(blob):
    return pickle.loads(blob)
def run(expr):
    return eval(expr)
"""

EXPECTED_KEYS = {"rule_id", "severity", "category", "description", "line", "match"}


def _by_rule_id(findings: list[dict]) -> dict[str, dict]:
    """Index findings by rule_id for easier assertions.

    If a rule fires more than once we keep the first hit — the tests below
    only assert per-rule behavior, not per-line.
    """
    out: dict[str, dict] = {}
    for f in findings:
        out.setdefault(f["rule_id"], f)
    return out


# ── Fixture tests ─────────────────────────────────────────────────────────

def test_flask_debug_and_bind_all():
    findings = run_static_scanner(FLASK_DEBUG_BIND_ALL)
    assert len(findings) >= 2, f"expected at least 2 findings, got {findings!r}"
    by_id = _by_rule_id(findings)

    assert "B201" in by_id, "bandit should flag flask debug=True"
    assert by_id["B201"]["severity"] == "high"       # HIGH sev + MEDIUM conf → high
    assert by_id["B201"]["category"] == "config"

    assert "B104" in by_id, "bandit should flag 0.0.0.0 bind"
    assert by_id["B104"]["severity"] == "medium"     # MEDIUM sev + MEDIUM conf → medium
    assert by_id["B104"]["category"] == "config"


def test_fstring_sql_cross_line_is_caught():
    """This is the key 'additive' case — the regex layer structurally misses
    an f-string SQL query defined on one line and execute()'d on another."""
    findings = run_static_scanner(FSTRING_SQL_CROSS_LINE)
    by_id = _by_rule_id(findings)

    assert "B608" in by_id, "bandit should flag f-string SQL construction"
    # MEDIUM severity + LOW confidence → low. Deliberately not "critical":
    # the referee's semantic layer should make the final call on whether the
    # interpolated value is actually user-controlled.
    assert by_id["B608"]["severity"] == "low"
    assert by_id["B608"]["category"] == "injection"


def test_md5_pickle_eval_severity_mapping():
    findings = run_static_scanner(MD5_PICKLE_EVAL)
    by_id = _by_rule_id(findings)

    # MD5: HIGH sev + HIGH conf → critical (weakest possible hash in security context)
    assert "B324" in by_id
    assert by_id["B324"]["severity"] == "critical"
    assert by_id["B324"]["category"] == "crypto"

    # pickle.loads: MEDIUM sev + HIGH conf → high
    assert "B301" in by_id
    assert by_id["B301"]["severity"] == "high"
    assert by_id["B301"]["category"] == "injection"

    # eval(): MEDIUM sev + HIGH conf → high
    assert "B307" in by_id
    assert by_id["B307"]["severity"] == "high"
    assert by_id["B307"]["category"] == "injection"

    # import pickle: LOW sev + HIGH conf → medium; category is config
    # (informational import flag, distinct from the pickle.loads call site)
    assert "B403" in by_id
    assert by_id["B403"]["severity"] == "medium"
    assert by_id["B403"]["category"] == "config"


# ── Empty input ───────────────────────────────────────────────────────────

def test_empty_input_returns_empty_list():
    assert run_static_scanner("") == []


def test_whitespace_only_input_returns_empty_list():
    assert run_static_scanner("   \n  \n\t\n") == []


# ── Graceful failure ──────────────────────────────────────────────────────

def test_missing_bandit_returns_empty_and_does_not_raise():
    """Simulate bandit not being installed in the venv (FileNotFoundError
    from the subprocess invocation)."""
    with patch("domains.code_security.static_scanner.subprocess.run",
               side_effect=FileNotFoundError("simulated missing bandit")):
        result = run_static_scanner("x = 1")
    assert result == []


def test_bandit_timeout_returns_empty_and_does_not_raise():
    with patch("domains.code_security.static_scanner.subprocess.run",
               side_effect=subprocess.TimeoutExpired(cmd="bandit", timeout=1)):
        result = run_static_scanner("x = 1")
    assert result == []


def test_bandit_garbage_output_returns_empty_and_does_not_raise():
    """Bandit runs but prints non-JSON — we must not raise."""
    class FakeProc:
        stdout = "not valid json at all {{{"
        stderr = ""
        returncode = 0

    with patch("domains.code_security.static_scanner.subprocess.run",
               return_value=FakeProc()):
        result = run_static_scanner("x = 1")
    assert result == []


def test_bandit_empty_stdout_returns_empty_and_does_not_raise():
    class FakeProc:
        stdout = ""
        stderr = "some error"
        returncode = 2

    with patch("domains.code_security.static_scanner.subprocess.run",
               return_value=FakeProc()):
        result = run_static_scanner("x = 1")
    assert result == []


# ── Shape test ────────────────────────────────────────────────────────────

def test_all_findings_have_complete_finding_shape():
    """Every finding dict must have exactly the 6 keys of detector.Finding.to_dict()."""
    all_findings = []
    for code in (FLASK_DEBUG_BIND_ALL, FSTRING_SQL_CROSS_LINE, MD5_PICKLE_EVAL):
        all_findings.extend(run_static_scanner(code))

    assert len(all_findings) > 0, "sanity: fixtures should produce findings"

    for f in all_findings:
        assert set(f.keys()) == EXPECTED_KEYS, (
            f"finding has wrong keys: got {set(f.keys())}, want {EXPECTED_KEYS}"
        )
        assert isinstance(f["rule_id"], str) and f["rule_id"]
        assert f["severity"] in {"critical", "high", "medium", "low", "info"}
        assert f["category"] in {"injection", "auth", "crypto", "secrets", "resource", "config"}
        assert isinstance(f["description"], str)
        assert f["line"] is None or isinstance(f["line"], int)
        assert isinstance(f["match"], str)
        assert len(f["match"]) <= 120


# ── Helper unit tests ─────────────────────────────────────────────────────

@pytest.mark.parametrize("sev,conf,expected", [
    ("HIGH",   "HIGH",   "critical"),
    ("HIGH",   "MEDIUM", "high"),
    ("HIGH",   "LOW",    "medium"),
    ("MEDIUM", "HIGH",   "high"),
    ("MEDIUM", "MEDIUM", "medium"),
    ("MEDIUM", "LOW",    "low"),
    ("LOW",    "HIGH",   "medium"),
    ("LOW",    "MEDIUM", "low"),
    ("LOW",    "LOW",    "low"),
    # Case insensitivity + missing values fall back gracefully
    ("high",   "high",   "critical"),
    (None,     None,     "low"),
])
def test_severity_matrix(sev, conf, expected):
    assert _map_severity(sev, conf) == expected


@pytest.mark.parametrize("test_name,test_id,expected", [
    # Explicit test_id map
    ("blacklist",                       "B301", "injection"),  # pickle
    ("blacklist",                       "B307", "injection"),  # eval
    ("blacklist",                       "B311", "crypto"),     # random
    ("blacklist",                       "B403", "config"),     # pickle import (informational)
    ("hardcoded_bind_all_interfaces",   "B104", "config"),
    ("flask_debug_true",                "B201", "config"),
    ("hardcoded_sql_expressions",       "B608", "injection"),
    ("hashlib_new_insecure_functions",  "B324", "crypto"),
    # Keyword fallback for unmapped IDs
    ("request_with_no_cert_validation", "B999", "crypto"),    # unknown id, matches 'cert'
    # Prefix range fallback
    ("",                                "B650", "injection"), # B6xx
    ("",                                "B350", "crypto"),    # B3xx
    ("",                                "B220", "config"),    # B2xx
    ("",                                "B150", "resource"),  # B1xx
    # Catch-all
    ("",                                "",     "config"),
])
def test_category_mapping(test_name, test_id, expected):
    assert _map_category(test_name, test_id) == expected


def test_severity_score_matches_detector_weights():
    """Same scale as detector.severity_score: 25/15/8/3/1 with a 100 cap."""
    findings = [
        {"severity": "critical"},
        {"severity": "high"},
        {"severity": "medium"},
        {"severity": "low"},
        {"severity": "info"},
    ]
    assert severity_score(findings) == 25 + 15 + 8 + 3 + 1  # 52

    capped = [{"severity": "critical"}] * 10  # 250 raw, capped to 100
    assert severity_score(capped) == 100

    assert severity_score([]) == 0
