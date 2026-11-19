from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest
import redis
from asgiref.sync import async_to_sync
from django.core.cache import cache

from overbae.core import llms
from overbae.core.model_registry import ROLE_CHAINS, Role
from overbae.innermind import replay
from overbae.innermind.evals import JevEvaluator
from overbae.models import EvalMatrix, Span
from overbae.services.codebase import task_type

pytestmark = pytest.mark.covers("llm-gateway")

REDIS_URL = "redis://localhost:6379/15"

LOCAL_MODELS = {"local/decision", "recursivesi/flint"}


def _capability():
    return SimpleNamespace(
        id=uuid.uuid4(),
        project_id=uuid.uuid4(),
        description="Route support requests to the appropriate team.",
        decision_logic="Choose the team from the request's subject.",
        policy_markdown="Return exactly one team.",
        model="",
        input_schema={},
        output_fields={},
        improvement_metadata={"system_prompt": "Classify the request into one support queue."},
    )


@pytest.fixture
def redis_cache(settings, monkeypatch):
    monkeypatch.setenv("JEV_REQUESTS_PER_MINUTE", "10000000")
    monkeypatch.setenv("JEV_TOKENS_PER_SECOND", "10000000")
    settings.CACHES = {
        "default": {
            "BACKEND": "django.core.cache.backends.redis.RedisCache",
            "LOCATION": REDIS_URL,
            "KEY_PREFIX": "overmind",
        }
    }
    return REDIS_URL


def test_every_request_names_a_local_model_and_explain_sends_none(fake_llm, redis_cache):
    cache.clear()
    assert ROLE_CHAINS[Role.EXPLAIN] == ()
    fake_llm.decide = lambda key, question, state: "no"
    fake_llm.on(
        lambda r: r.system.startswith("Classify the task an AI capability"),
        '{"task_type":"classification"}',
    )
    span = Span(
        span_id="span",
        trace_id="trace",
        attributes={"input": "Where is my order?", "output": "I like turtles."},
        eval_attributes={},
    )
    row = EvalMatrix(name="relevant", type=EvalMatrix.Type.LLM, source="Answers the question")

    # A low score is what asks for an explanation; the run naming a model does not change that.
    metric = async_to_sync(JevEvaluator("project", explain_model="gpt-6-luna").score)(
        row, span, [span]
    )
    assert metric.score == 0
    assert metric.reason == ""

    assert task_type.classify_capability_task(_capability()) == "classification"

    tools = SimpleNamespace(tool_definitions=lambda: [], execute=lambda *a, **k: None)
    result = replay.run_capability(
        input_messages=[{"role": "user", "content": "hello"}], tool_provider=tools, model="flint"
    )
    assert not result.error

    sent = [request.model for request in fake_llm.requests]
    sent += [decision["model"] for decision in fake_llm.decisions]
    assert set(sent) == LOCAL_MODELS


def test_a_flint_call_waits_for_the_stream_and_gives_up(fake_llm, redis_cache):
    client = redis.Redis.from_url(redis_cache)
    client.set("overmind:llm:flint:lock", "another-caller", px=60_000)
    try:
        with pytest.raises(llms.LLMBusyError):
            llms.call_llm("hello", model="flint", retry_deadline=0)
        assert fake_llm.requests == []

        client.delete("overmind:llm:flint:lock")
        llms.call_llm("hello", model="flint", retry_deadline=0)
        assert [request.model for request in fake_llm.requests] == ["recursivesi/flint"]
        assert client.get("overmind:llm:flint:lock") is None
    finally:
        client.delete("overmind:llm:flint:lock")
