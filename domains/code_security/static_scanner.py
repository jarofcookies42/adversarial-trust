"""Static analysis scanner (AST-based) — third evaluation layer for code security.

Wraps bandit so its findings slot into the same Finding shape produced by
detector.py (the regex layer). This lets graph.py treat both deterministic
layers uniformly without caring which one produced a given finding.

The three layers of the code_security domain are deliberately kept
independent and NOT merged:

  1. detector.py (regex)    — line-local textual patterns
  2. static_scanner (bandit) — AST-level structural patterns, cross-line
  3. LLM referee           — semantic / dataflow / contextual judgment

Each catches a different slice of the attack surface. Measuring them
separately (per Phase 1's comparison) is the whole point — so this module
must never fold into detector.py.

Invoke bandit via `sys.executable -m bandit`, so the subprocess inherits
the venv's bandit installation (uv-managed .venv or otherwise). No
reliance on PATH.

All failure modes return an empty list. This function must not raise —
the graph relies on it for best-effort augmentation only.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── Severity mapping ──────────────────────────────────────────────────────
#
# Bandit reports two axes per finding:
#   issue_severity:   LOW / MEDIUM / HIGH  — potential impact if real
#   issue_confidence: LOW / MEDIUM / HIGH  — how sure bandit is it's a TP
#
# We collapse them to detector.py's single-axis scale
# (critical / high / medium / low / info). Confidence shifts severity by
# one tier — HIGH bumps up, MEDIUM leaves as-is, LOW bumps down. This
# matches the pattern specified: HIGH+HIGH = critical, HIGH+MEDIUM = high,
# etc. The full matrix is explicit below to avoid any ambiguity.

_SEVERITY_MATRIX: dict[tuple[str, str], str] = {
    # (bandit_severity, bandit_confidence): detector_severity
    ("HIGH",   "HIGH"):   "critical",
    ("HIGH",   "MEDIUM"): "high",
    ("HIGH",   "LOW"):    "medium",
    ("MEDIUM", "HIGH"):   "high",
    ("MEDIUM", "MEDIUM"): "medium",
    ("MEDIUM", "LOW"):    "low",
    ("LOW",    "HIGH"):   "medium",
    ("LOW",    "MEDIUM"): "low",
    ("LOW",    "LOW"):    "low",
}


def _map_severity(bandit_severity: str | None, bandit_confidence: str | None) -> str:
    """Combine bandit severity + confidence into detector's severity scale."""
    key = ((bandit_severity or "LOW").upper(), (bandit_confidence or "LOW").upper())
    return _SEVERITY_MATRIX.get(key, "low")


# ── Category mapping ──────────────────────────────────────────────────────
#
# detector.py uses categories: injection, auth, crypto, secrets, resource, config.
# We map bandit findings into the same set. Resolution order:
#   1. Explicit test_id lookup (most reliable — bandit's IDs are stable).
#   2. Keyword search against test_name (covers newer rules we haven't mapped).
#   3. Test-ID prefix range fallback (B6xx/B7xx injection, B3xx/B5xx crypto, etc).
#   4. Catch-all "config".
#
# Bandit's B3xx "blacklist" tests (pickle, eval, marshal, random, etc.) all
# share test_name="blacklist", so the keyword-only approach misses them —
# the explicit test_id map is required.

_TEST_ID_CATEGORIES: dict[str, str] = {
    # B1xx — misc
    "B101": "resource",  # assert_used
    "B102": "injection", # exec_used
    "B103": "resource",  # set_bad_file_permissions
    "B104": "config",    # hardcoded_bind_all_interfaces
    "B105": "secrets",   # hardcoded_password_string
    "B106": "secrets",   # hardcoded_password_funcarg
    "B107": "secrets",   # hardcoded_password_default
    "B108": "resource",  # hardcoded_tmp_directory
    "B110": "resource",  # try_except_pass
    "B112": "resource",  # try_except_continue
    # B2xx — framework configuration
    "B201": "config",    # flask_debug_true
    "B202": "resource",  # tarfile_unsafe_members
    # B3xx — blacklisted calls / crypto
    "B301": "injection", # pickle
    "B302": "injection", # marshal
    "B303": "crypto",    # md5 (historical)
    "B304": "crypto",    # insecure ciphers
    "B305": "crypto",    # cipher_modes
    "B306": "resource",  # mktemp_q
    "B307": "injection", # eval
    "B308": "injection", # mark_safe
    "B310": "injection", # urllib_urlopen
    "B311": "crypto",    # random (non-cryptographic)
    "B312": "config",    # telnetlib
    "B321": "config",    # ftplib
    "B323": "crypto",    # unverified_context
    "B324": "crypto",    # hashlib_new_insecure_functions
    # B5xx — SSL/TLS
    "B501": "crypto",    # request_with_no_cert_validation
    "B502": "crypto",    # ssl_with_bad_version
    "B503": "crypto",    # ssl_with_bad_defaults
    "B504": "crypto",    # ssl_with_no_version
    "B505": "crypto",    # weak_cryptographic_key
    "B506": "injection", # yaml_load
    "B507": "crypto",    # ssh_no_host_key_verification
    # B6xx — injection
    "B601": "injection", # paramiko_calls
    "B602": "injection", # subprocess_popen_with_shell_equals_true
    "B603": "injection", # subprocess_without_shell_equals_true
    "B604": "injection", # any_other_function_with_shell_equals_true
    "B605": "injection", # start_process_with_a_shell
    "B606": "injection", # start_process_with_no_shell
    "B607": "injection", # start_process_with_partial_path
    "B608": "injection", # hardcoded_sql_expressions
    "B609": "injection", # linux_commands_wildcard_injection
    "B610": "injection", # django_extra_used
    "B611": "injection", # django_rawsql_used
    # B7xx — templates / XSS
    "B701": "config",    # jinja2_autoescape_false
    "B702": "config",    # use_of_mako_templates
    "B703": "injection", # django_mark_safe
    "B704": "injection", # markupsafe_markup_xss
}

_CATEGORY_KEYWORDS: list[tuple[str, str]] = [
    ("sql",        "injection"),
    ("inject",     "injection"),
    ("shell",      "injection"),
    ("subprocess", "injection"),
    ("eval",       "injection"),
    ("pickle",     "injection"),
    ("marshal",    "injection"),
    ("yaml",       "injection"),
    ("xml",        "injection"),
    ("hash",       "crypto"),
    ("md5",        "crypto"),
    ("sha1",       "crypto"),
    ("ssl",        "crypto"),
    ("tls",        "crypto"),
    ("cert",       "crypto"),
    ("cipher",     "crypto"),
    ("random",     "crypto"),
    ("password",   "secrets"),
    ("token",      "secrets"),
    ("secret",     "secrets"),
    ("tmp",        "resource"),
    ("tempfile",   "resource"),
    ("assert",     "resource"),
    ("flask",      "config"),
    ("django",     "config"),
    ("debug",      "config"),
    ("bind",       "config"),
    ("cors",       "config"),
]


def _map_category(test_name: str | None, test_id: str | None = None) -> str:
    """Derive detector-scale category for a bandit finding."""
    tid = (test_id or "").strip()
    if tid in _TEST_ID_CATEGORIES:
        return _TEST_ID_CATEGORIES[tid]

    name = (test_name or "").lower()
    for kw, cat in _CATEGORY_KEYWORDS:
        if kw in name:
            return cat

    # Prefix-range fallback for unmapped bandit IDs
    if tid.startswith(("B6", "B7")):
        return "injection"
    if tid.startswith(("B3", "B5")):
        return "crypto"
    if tid.startswith("B2"):
        return "config"
    if tid.startswith("B1"):
        return "resource"
    return "config"


# ── Scanner ───────────────────────────────────────────────────────────────

def run_static_scanner(code: str, timeout: float = 30.0) -> list[dict[str, Any]]:
    """Run bandit against `code`, return findings in detector's Finding shape.

    Returns:
        List of dicts with keys rule_id, severity, category, description,
        line, match — matching detector.Finding.to_dict() exactly.

    Never raises. Returns [] on any failure (bandit not installed, timeout,
    subprocess error, unparseable output) and logs a warning.
    """
    with tempfile.NamedTemporaryFile(
        "w", suffix=".py", delete=False, encoding="utf-8"
    ) as tf:
        tf.write(code)
        tmp_path = tf.name

    try:
        try:
            proc = subprocess.run(
                [
                    sys.executable, "-m", "bandit",
                    "-f", "json",
                    "-q",
                    "--exit-zero",  # exit 0 regardless; detect errors via stdout
                    tmp_path,
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            logger.warning("static_scanner: bandit timed out after %ss", timeout)
            return []
        except (FileNotFoundError, OSError) as e:
            logger.warning("static_scanner: bandit invocation failed: %s", e)
            return []
    finally:
        try:
            Path(tmp_path).unlink()
        except OSError:
            pass

    if not proc.stdout.strip():
        logger.warning(
            "static_scanner: bandit returned empty stdout (stderr: %s)",
            (proc.stderr or "").strip()[:200],
        )
        return []

    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        logger.warning("static_scanner: could not parse bandit JSON output: %s", e)
        return []

    raw_results = payload.get("results", []) or []
    findings: list[dict[str, Any]] = []
    for r in raw_results:
        try:
            snippet = (r.get("code") or "").strip().replace("\n", " ").replace("\r", "")
            findings.append({
                "rule_id":     r.get("test_id", "") or "",
                "severity":    _map_severity(r.get("issue_severity"), r.get("issue_confidence")),
                "category":    _map_category(r.get("test_name"), r.get("test_id")),
                "description": (r.get("issue_text") or r.get("test_name") or "").strip(),
                "line":        r.get("line_number"),
                "match":       snippet[:120],
            })
        except Exception as e:  # pragma: no cover — defensive
            logger.warning("static_scanner: malformed bandit result skipped: %s", e)
            continue

    return findings


def severity_score(findings: list[dict]) -> int:
    """Aggregate severity score (0-100). Same weights as detector.severity_score."""
    weights = {"critical": 25, "high": 15, "medium": 8, "low": 3, "info": 1}
    total = sum(weights.get(f.get("severity", "info"), 1) for f in findings)
    return min(total, 100)


# ── Self-test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Run via: uv run python domains/code_security/static_scanner.py
    samples = {
        "flask_debug_bind_all": """\
from flask import Flask
app = Flask(__name__)
app.run(debug=True, host="0.0.0.0", port=5000)
""",
        "fstring_sql_cross_line": """\
import psycopg2
def find_user(conn, name):
    query = f"SELECT * FROM users WHERE name = '{name}'"
    with conn.cursor() as cur:
        cur.execute(query)
        return cur.fetchall()
""",
        "md5_pickle_eval": """\
import hashlib, pickle
def h(u):
    return hashlib.md5(u.encode()).hexdigest()
def load(blob):
    return pickle.loads(blob)
def run(expr):
    return eval(expr)
""",
    }

    for name, code in samples.items():
        findings = run_static_scanner(code)
        score = severity_score(findings)
        print(f"\n=== {name}  ({len(findings)} findings, severity_score={score}) ===")
        for f in findings:
            print(
                f"  [{f['severity']:<8}] {f['rule_id']:<5} {f['category']:<10} "
                f"L{f['line']:<4} {f['description'][:80]}"
            )
