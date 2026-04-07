"""Programmatic vulnerability detection for code security domain.

This is the second layer of the dual-layer evaluation system.
The LLM judge catches semantic/contextual issues; this catches
pattern-based issues the LLM might miss (or confirms what it found).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Finding:
    """A single programmatic detection result."""
    rule_id: str
    severity: str  # critical, high, medium, low, info
    category: str  # injection, auth, crypto, secrets, resource, config
    description: str
    line: int | None = None
    match: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "severity": self.severity,
            "category": self.category,
            "description": self.description,
            "line": self.line,
            "match": self.match,
        }


# Each rule: (rule_id, severity, category, pattern, description)
_RULES: list[tuple[str, str, str, str, str]] = [
    # Injection
    ("INJ-001", "critical", "injection",
     r"""(execute|cursor\.execute|raw|rawsql)\s*\(.*(%s|\.format|f['"])""",
     "Possible SQL injection via string formatting"),
    ("INJ-002", "critical", "injection",
     r"""(subprocess\.(call|run|Popen)|os\.system|os\.popen)\s*\(.*(%s|\.format|f['"]|\+)""",
     "Possible command injection via string concatenation"),
    ("INJ-003", "high", "injection",
     r"""eval\s*\(|exec\s*\(""",
     "Use of eval/exec — potential code injection"),
    ("INJ-004", "high", "injection",
     r"""__import__\s*\(""",
     "Dynamic import — potential code injection"),

    # Authentication / Authorization
    ("AUTH-001", "critical", "auth",
     r"""(password|passwd|pwd)\s*==\s*['"]""",
     "Hardcoded password comparison"),
    ("AUTH-002", "high", "auth",
     r"""(verify|check).*=\s*False""",
     "Security verification explicitly disabled"),
    ("AUTH-003", "medium", "auth",
     r"""(jwt|token).*algorithm\s*=\s*['"]none['"]""",
     "JWT with 'none' algorithm"),
    ("AUTH-004", "high", "auth",
     r"""SECRET_KEY\s*=\s*['"][^'"]{0,20}['"]""",
     "Short or hardcoded secret key"),

    # Cryptography
    ("CRYPTO-001", "high", "crypto",
     r"""(md5|sha1)\s*\(""",
     "Weak hash function (MD5/SHA1)"),
    ("CRYPTO-002", "critical", "crypto",
     r"""\b(DES|RC4|Blowfish)\b""",
     "Weak or deprecated cipher"),
    ("CRYPTO-003", "medium", "crypto",
     r"""random\.(random|randint|choice|randrange)""",
     "Non-cryptographic random for potentially sensitive use"),

    # Secrets / Credentials
    ("SEC-001", "critical", "secrets",
     r"""(api[_-]?key|apikey|secret[_-]?key|access[_-]?token|auth[_-]?token)\s*=\s*['"][A-Za-z0-9+/=_-]{8,}['"]""",
     "Hardcoded API key or secret token"),
    ("SEC-002", "high", "secrets",
     r"""(password|passwd|pwd)\s*=\s*['"][^'"]+['"]""",
     "Hardcoded password string"),
    ("SEC-003", "medium", "secrets",
     r"""(DB_|DATABASE_|MONGO_|REDIS_)(URL|URI|HOST|PASSWORD)\s*=\s*['"]""",
     "Hardcoded database credential"),

    # Resource / Error handling
    ("RES-001", "medium", "resource",
     r"""open\s*\([^)]+\)(?!.*\bwith\b)""",
     "File opened without context manager (potential resource leak)"),
    ("RES-002", "medium", "resource",
     r"""except\s*:\s*$|except\s+Exception\s*:\s*(pass|\.\.\.)\s*$""",
     "Bare except or swallowed exception"),
    ("RES-003", "low", "resource",
     r"""# ?TODO|# ?FIXME|# ?HACK|# ?XXX""",
     "TODO/FIXME marker — potential incomplete implementation"),

    # Configuration
    ("CFG-001", "high", "config",
     r"""DEBUG\s*=\s*True|debug\s*=\s*True""",
     "Debug mode enabled"),
    ("CFG-002", "medium", "config",
     r"""CORS.*[\*]|allow_all|AllowAny|cors_allowed_origins\s*=\s*\[?\s*['\"]\*""",
     "Overly permissive CORS configuration"),
    ("CFG-003", "medium", "config",
     r"""(verify_ssl|check_hostname|CERT_NONE|verify\s*=\s*False)""",
     "SSL/TLS verification disabled"),

    # Deserialization
    ("DESER-001", "critical", "injection",
     r"""(pickle\.loads?|yaml\.load\s*\([^)]*(?!Loader))""",
     "Unsafe deserialization (pickle/yaml.load without SafeLoader)"),
]


def run_detector(code: str) -> list[dict[str, Any]]:
    """Run all programmatic detection rules against the given code.

    Returns a list of finding dicts, each with rule_id, severity,
    category, description, line number, and matching text.
    """
    findings: list[Finding] = []
    lines = code.split("\n")

    # Rules that must be matched case-sensitively
    case_sensitive_rules = {"CRYPTO-002"}

    for rule_id, severity, category, pattern, description in _RULES:
        try:
            flags = 0 if rule_id in case_sensitive_rules else re.IGNORECASE
            regex = re.compile(pattern, flags)
        except re.error:
            continue

        for i, line in enumerate(lines, start=1):
            # Skip comment-only lines
            stripped = line.strip()
            if stripped.startswith("#") and rule_id != "RES-003":
                continue
            for m in regex.finditer(line):
                findings.append(Finding(
                    rule_id=rule_id,
                    severity=severity,
                    category=category,
                    description=description,
                    line=i,
                    match=m.group(0)[:120],
                ))

    # Deduplicate by (rule_id, line)
    seen: set[tuple[str, int | None]] = set()
    unique: list[Finding] = []
    for f in findings:
        key = (f.rule_id, f.line)
        if key not in seen:
            seen.add(key)
            unique.append(f)

    return [f.to_dict() for f in unique]


def severity_score(findings: list[dict]) -> int:
    """Compute an aggregate severity score (0-100) from findings."""
    weights = {"critical": 25, "high": 15, "medium": 8, "low": 3, "info": 1}
    total = sum(weights.get(f.get("severity", "info"), 1) for f in findings)
    return min(total, 100)
