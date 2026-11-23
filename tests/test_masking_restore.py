from __future__ import annotations

import json

import pytest
from fakes import masking as fake_ner

from overbae.services.masking import MaskingError, derive, mask_state

pytestmark = pytest.mark.covers("masking-restore")


@pytest.fixture(autouse=True)
def _ner(settings):
    settings.MASKING_NER_BACKEND = fake_ner.SPEC
    settings.MASKING_TIMEOUT_SECONDS = 5
    settings.MASKING_MAX_BYTES = 24_000
    fake_ner.NAMES[:] = ["Ravi Kumar"]
    yield
    fake_ner.NAMES[:] = []


def test_an_explanation_is_restored_from_the_map_of_the_call_that_produced_it():
    payload = mask_state({"output": "Ravi Kumar paid Rs. 18,000 and called 9876543210 today"})
    explanation = "<PERSON_1> paid <AMOUNT_1>; <PHONE_1> is repeated, and <PERSON_9> is not known."

    assert payload.restore(explanation) == (
        "Ravi Kumar paid Rs. 18,000; 9876543210 is repeated, and <PERSON_9> is not known."
    )


def test_a_derived_prompt_keeps_the_same_map_and_never_holds_a_raw_value():
    payload = mask_state({"output": "Ravi Kumar called 9876543210 about the delivery"})
    prompt = derive(payload, "An evaluator scored this 0.00 out of 1.\n\n")

    assert prompt.text.startswith("An evaluator scored")
    assert (
        json.loads(prompt.text.split("\n\n", 1)[1])["output"]
        == "<PERSON_1> called <PHONE_1> about the delivery"
    )
    assert prompt.restore("<PERSON_1>") == "Ravi Kumar"
    assert "9876543210" not in repr(prompt) and "Ravi" not in repr(prompt)


def test_wording_that_carries_a_value_is_refused():
    payload = mask_state({"output": "hello"})

    with pytest.raises(MaskingError) as refused:
        derive(payload, "The customer at 9876543210 scored this 0.00.\n\n")

    assert refused.value.reason == "residual"
