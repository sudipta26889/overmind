import os
import subprocess
import sys

import pytest

from overbae.core.model_registry import (
    CATALOG,
    OPENROUTER_MODEL_SLUGS,
    ROLE_CHAINS,
    Role,
    RoleOffError,
    TaskType,
    _role_chain,
    decision_model,
    default_backtest_models,
    default_judge_model,
    inference_models,
    is_decision_model,
    judge_picker_models,
    model_chain,
    model_defaults,
    openrouter_configured,
    pricing_slug,
    resolve_model,
    resolve_openrouter_slug,
    role_enabled,
)

pytestmark = pytest.mark.covers("llm-gateway")


@pytest.fixture
def no_keys(monkeypatch):
    for env in (
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
    ):
        monkeypatch.delenv(env, raising=False)


def test_availability_needs_the_openrouter_key(no_keys, monkeypatch):
    assert openrouter_configured() is False
    monkeypatch.setenv("OPENAI_API_KEY", "sk-direct")
    assert openrouter_configured() is False
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    assert openrouter_configured() is True


def test_resolve_model_returns_the_head_of_the_chain(no_keys, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    assert resolve_model(TaskType.JUDGE_SCORING) == "flint"
    assert resolve_model(TaskType.DEFAULT) == "flint"


def test_resolve_model_raises_without_a_key(no_keys):
    with pytest.raises(RuntimeError, match="No LLM API key"):
        resolve_model(TaskType.JUDGE_SCORING)


def test_an_empty_chain_turns_its_role_off(monkeypatch):
    monkeypatch.setitem(ROLE_CHAINS, Role.EXPLAIN, ())
    assert role_enabled(Role.EXPLAIN) is False
    assert role_enabled(Role.FAST) is True
    with pytest.raises(RoleOffError):
        resolve_model(TaskType.EVAL_EXPLANATION)
    assert model_chain(TaskType.EVAL_EXPLANATION) == []


def test_unknown_task_falls_back_to_the_default_chain():
    assert model_chain("not-a-task") == model_chain(TaskType.DEFAULT)


def test_every_chain_name_is_routable():
    for chain in ROLE_CHAINS.values():
        assert all(model in OPENROUTER_MODEL_SLUGS for model in chain)


def test_chains_come_only_from_the_environment(monkeypatch):
    monkeypatch.delenv("LLM_FAST_CHAIN", raising=False)
    assert _role_chain("LLM_FAST_CHAIN") == ()
    monkeypatch.setenv("LLM_FAST_CHAIN", "flint, gpt-6-luna")
    assert _role_chain("LLM_FAST_CHAIN") == ("flint", "gpt-6-luna")
    monkeypatch.setenv("LLM_FAST_CHAIN", "no-such-model")
    with pytest.raises(ValueError, match="outside the catalog"):
        _role_chain("LLM_FAST_CHAIN")


def test_defaults_come_from_the_chains():
    assert default_judge_model() == "flint"
    assert judge_picker_models() == ["flint"]
    assert default_backtest_models() == ["recursivesi/flint"]


def test_defaults_are_empty_when_every_chain_is_empty(monkeypatch):
    for role in Role:
        monkeypatch.setitem(ROLE_CHAINS, role, ())
    assert model_defaults() == {"judge_model": "", "judge_models": [], "backtest_models": []}


def test_the_decision_model_is_named_by_the_environment(monkeypatch):
    assert decision_model().slug == "local/decision"
    assert is_decision_model("local/decision")
    assert not is_decision_model("recursivesi/flint")
    monkeypatch.setenv("JEV_MODEL", "")
    with pytest.raises(ValueError, match="JEV_MODEL is required"):
        decision_model()
    monkeypatch.setenv("JEV_MODEL", "typesafe/jev-1.14")
    with pytest.raises(ValueError, match="not one of"):
        decision_model()


def test_flint_stays_out_of_the_inference_library():
    assert "flint" not in {m.name for m in inference_models()}


def test_the_app_starts_with_every_chain_empty():
    environment = {
        **{key: value for key, value in os.environ.items() if not key.startswith("LLM_")},
        "LLM_FAST_CHAIN": "",
        "LLM_AGENT_CHAIN": "",
        "LLM_EXPLAIN_CHAIN": "",
    }
    code = (
        "import django; django.setup(); import overbae.api.serializers; "
        "from overbae.core.model_registry import model_defaults; "
        "print(model_defaults())"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], env=environment, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert "'judge_models': []" in result.stdout


def test_catalog_names_are_unique_and_slugs_are_vendor_qualified():
    names = [m.name for m in CATALOG]
    assert len(names) == len(set(names))
    assert all("/" in m.slug for m in CATALOG if m.slug)


def test_pricing_slug_covers_catalog_and_explicit_slugs():
    assert pricing_slug("claude-sonnet-5") == "anthropic/claude-sonnet-5"
    assert pricing_slug("openai/gpt-5.6-terra") == "openai/gpt-5.6-terra"
    assert pricing_slug("nope") is None


def test_resolve_openrouter_slug_qualifies_bare_vendor_names():
    assert resolve_openrouter_slug("claude-sonnet-5") == "anthropic/claude-sonnet-5"
    assert resolve_openrouter_slug("gpt-4o-mini") == "openai/gpt-4o-mini"
    assert resolve_openrouter_slug("acme/custom") == "acme/custom"
    assert resolve_openrouter_slug("mystery") == "mystery"


def test_gpt_6_luna_is_in_the_catalog_for_the_self_hosted_gateway(monkeypatch):
    # The self-hosted LiteLLM (db.lan) serves the fast tier as openai/gpt-6-luna, not gpt-5.6-luna.
    assert pricing_slug("gpt-6-luna") == "openai/gpt-6-luna"
    monkeypatch.setenv("LLM_EXPLAIN_CHAIN", "gpt-6-luna")
    assert _role_chain("LLM_EXPLAIN_CHAIN") == ("gpt-6-luna",)
