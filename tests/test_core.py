"""Basic tests for MAAT framework."""

from maat.config import MODELS, ModelConfig, Provider, TEMP_TARGET, TEMP_ATTACKER
from maat.state import CodeSecurityState, SecretExtractionState


def test_model_configs_exist():
    """Verify pre-built model configs are accessible."""
    assert "gemini-2.5-flash" in MODELS
    assert "mistral" in MODELS
    assert "haiku" in MODELS


def test_model_config_properties():
    """Verify ModelConfig properties work."""
    config = ModelConfig(Provider.GOOGLE, "gemini-2.5-flash", temperature=0.3)
    assert config.display_name == "google/gemini-2.5-flash"
    assert config.temperature == 0.3


def test_temperature_conventions():
    """Verify temperature conventions are sensible."""
    assert TEMP_TARGET < TEMP_ATTACKER
    assert TEMP_TARGET == 0.3
    assert TEMP_ATTACKER == 0.9


def test_code_security_state_fields():
    """Verify CodeSecurityState has expected fields."""
    state: CodeSecurityState = {
        "domain": "code_security",
        "task_spec": "implement auth",
        "generated_code": "print('hello')",
        "finder_issues": [],
        "adversary_challenges": [],
        "surviving_issues": [],
        "referee_verdict": {},
        "programmatic_findings": [],
        "conversation_history": [],
        "judge_evaluations": [],
        "round_number": 0,
        "max_rounds": 1,
        "metadata": {},
    }
    assert state["domain"] == "code_security"
    assert isinstance(state["finder_issues"], list)
