from __future__ import annotations

import json

import pytest
from fakes import masking as fake_ner

from overbae.services.masking import MaskedPayload, MaskingError, mask_state
from overbae.services.masking.checksums import verhoeff_valid

pytestmark = pytest.mark.covers("masking-detectors")


@pytest.fixture(autouse=True)
def _ner(settings):
    settings.MASKING_NER_BACKEND = fake_ner.SPEC
    settings.MASKING_TIMEOUT_SECONDS = 5
    settings.MASKING_MAX_BYTES = 24_000
    fake_ner.NAMES[:] = []
    yield
    fake_ner.NAMES[:] = []


def aadhaar(prefix="23456789012") -> str:
    digit = next(d for d in range(10) if verhoeff_valid([int(c) for c in prefix + str(d)]))
    return prefix + str(digit)


def masked(text: str, **state) -> str:
    return json.loads(mask_state({"output": text, **state}).text)["output"]


@pytest.mark.parametrize(
    ("value", "kind"),
    [
        ("9876543210", "PHONE"),
        ("+91 98765 43210", "PHONE"),
        ("ravi.kumar@example.com", "EMAIL"),
        ("ravi@okhdfcbank", "ACCOUNT"),
        ("ABCDE1234F", "ID"),
        ("HDFC0001234", "ACCOUNT"),
        ("22AAAAA0000A1Z5", "ID"),
        ("4111 1111 1111 1111", "ACCOUNT"),
        ("192.168.10.24", "ID"),
        ("https://pay.example.com/r?token=abc123", "URL"),
        ("123456", "ID"),
        ("98765432101234", "ID"),
    ],
)
def test_structured_identifiers_become_surrogates(value, kind):
    out = masked(f"Please look at {value} for me, thank you very much.")
    assert value not in out
    assert f"<{kind}_1>" in out


def test_an_aadhaar_number_with_a_valid_checksum_is_masked_in_any_grouping():
    number = aadhaar()
    for form in (
        number,
        f"{number[:4]} {number[4:8]} {number[8:]}",
        f"{number[:4]}-{number[4:8]}-{number[8:]}",
    ):
        assert "<ID_1>" in masked(f"Aadhaar {form} on file")


def test_any_run_of_six_digits_is_an_identifier_whatever_the_checksum_says():
    assert masked("code 482915 here") == "code <ID_1> here"
    assert masked("order 4829 here") == "order 4829 here"


@pytest.mark.parametrize(
    "text",
    [
        "pay ₹18,000 today",
        "Rs. 2.5 lakh balance",
        "INR 1,20,000",
        "salary is 45000",
        "3 crore rupees",
        "$ 250",
        "18000 EMI",
    ],
)
def test_amounts_are_replaced(text):
    out = masked(text)
    assert "<AMOUNT_1>" in out
    assert not any(ch.isdigit() for ch in out.replace("_1", ""))


@pytest.mark.parametrize(
    "text",
    [
        "due on 5th March 2026",
        "on 05/03/2026",
        "2026-03-05",
        "March 5, 2026",
        "at 10:30 am",
        "come on the 12th",
        "৫ তারিখ",
    ],
)
def test_dates_and_times_are_replaced(text):
    assert "<DATE_1>" in masked(text)


@pytest.mark.parametrize(
    "text",
    ["took 500 mg", "weighs 72 kg", "BP 140/90", "sugar 240", "HbA1c 8.1", "glucose 7.8 mmol/l"],
)
def test_measured_values_are_replaced(text):
    out = masked(text)
    assert "<VALUE_1>" in out
    assert "240" not in out and "140" not in out and "500" not in out


def test_devanagari_and_bengali_numerals_count_as_digits():
    assert "<ID_1>" in masked("नंबर ९८७६५४३२१०१२ है")
    assert "<ID_1>" in masked("নম্বর ৯৮৭৬৫৪৩২১০১২ আছে")


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("He has diabetes and takes metformin.", "CONDITION"),
        ("मुझे मधुमेह है", "CONDITION"),
        ("amar jor ar kashi ache", "CONDITION"),
        ("মেটফর্মিন খাচ্ছি", "MEDICATION"),
        ("my HbA1c report", "TEST"),
        ("loan from HDFC Bank", "ORG"),
        ("maa ko dawai chahiye", "RELATION"),
        ("দিদি ফোন করেছিল", "RELATION"),
    ],
)
def test_the_lexicon_replaces_health_finance_and_relation_terms(text, kind):
    assert f"<{kind}_1>" in masked(text)


def test_the_same_value_gets_the_same_surrogate_and_a_new_one_gets_the_next():
    out = masked("call 9876543210 or 9876543210 or 9123456780")
    assert out == "call <PHONE_1> or <PHONE_1> or <PHONE_2>"


def test_names_come_from_the_model_and_an_absent_model_fails_closed(settings):
    fake_ner.NAMES[:] = ["Ravi Kumar"]
    assert masked("Message from Ravi Kumar about the delivery of the parcel today") == (
        "Message from <PERSON_1> about the delivery of the parcel today"
    )
    settings.MASKING_NER_BACKEND = ""
    settings.MASKING_MODEL_DIR = "/nonexistent"
    with pytest.raises(MaskingError) as refused:
        masked("Message from Ravi Kumar about the delivery of the parcel today")
    assert refused.value.reason == "model_not_loaded"


def test_a_crashing_model_fails_closed(settings):
    settings.MASKING_NER_BACKEND = "fakes.masking:Broken"
    with pytest.raises(MaskingError) as refused:
        masked("a sentence that is long enough to be read by the model, surely")
    assert refused.value.reason == "ner_error"


def test_a_short_sentence_is_read_by_the_model_but_a_short_token_is_not(settings):
    fake_ner.NAMES[:] = ["Ravi Kumar"]
    assert masked("Help Ravi Kumar now") == "Help <PERSON_1> now"
    settings.MASKING_NER_BACKEND = "fakes.masking:Broken"
    assert masked("yes") == "yes"
    assert masked("lookup_order") == "lookup_order"
    with pytest.raises(MaskingError):
        masked("Help me")


def test_a_sensitive_key_masks_its_value_whatever_it_says():
    state = {
        "input": json.dumps(
            {"patient_name": "Asha", "contactPhone": "n/a", "amount": 5, "note": "fine"}
        )
    }
    out = json.loads(json.loads(mask_state(state).text)["input"])
    assert out["patient_name"].startswith("<PERSON_")
    assert out["contactPhone"].startswith("<PHONE_")
    assert out["amount"].startswith("<AMOUNT_")
    assert out["note"] == "fine"


def test_tool_names_survive_and_tool_arguments_are_masked():
    call = {
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "lookup_order",
                    "arguments": json.dumps({"phone": "9876543210", "order": "short"}),
                },
            }
        ]
    }
    out = json.loads(mask_state({"output": json.dumps(call)}).text)["output"]
    function = json.loads(out)["tool_calls"][0]["function"]
    assert function["name"] == "lookup_order"
    assert "9876543210" not in function["arguments"]
    assert "<PHONE_1>" in function["arguments"]


def test_json_inside_json_is_decoded_masked_and_encoded_again():
    inner = json.dumps({"a": json.dumps({"mobile": "9876543210"})})
    out = masked(inner)
    assert "9876543210" not in out
    assert json.loads(json.loads(out)["a"])["mobile"] == "<PHONE_1>"


def test_attachments_become_a_marker_and_unknown_parts_fail_closed():
    state = {
        "input": [
            {"role": "user", "content": [{"type": "text", "text": "see 9876543210"}]},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]},
            {"role": "user", "content": "data:image/png;base64,AAAA"},
        ]
    }
    out = json.loads(mask_state(state).text)["input"]
    assert out[0]["content"][0]["text"] == "see <PHONE_1>"
    assert out[1]["content"] == ["<ATTACHMENT>"]
    assert out[2]["content"] == "<ATTACHMENT>"
    with pytest.raises(MaskingError) as refused:
        mask_state({"input": [{"content": [{"type": "video", "url": "x"}]}]})
    assert refused.value.reason == "non_text_part"


def test_existing_surrogate_looking_text_is_escaped_and_restores_exactly():
    payload = mask_state({"output": "the user typed <PERSON_1> and 9876543210"})
    out = json.loads(payload.text)["output"]
    assert out == "the user typed <ESC:PERSON_1> and <PHONE_1>"
    assert payload.restore(out) == "the user typed <PERSON_1> and 9876543210"


def test_the_size_cap_and_the_time_limit_fail_closed(settings):
    settings.MASKING_MAX_BYTES = 50
    with pytest.raises(MaskingError) as too_large:
        masked("x" * 100)
    assert too_large.value.reason == "too_large"
    settings.MASKING_MAX_BYTES = 24_000
    settings.MASKING_TIMEOUT_SECONDS = 0
    with pytest.raises(MaskingError) as slow:
        masked("hello")
    assert slow.value.reason == "timeout"


def test_the_residual_check_refuses_anything_still_visible_in_the_final_string():
    def leaky(state):
        return {**state, "output": state["output"] + " call 9876543210"}

    with pytest.raises(MaskingError) as refused:
        mask_state({"output": "hello there"}, after_mask=leaky)
    assert refused.value.reason == "residual"
    assert "PHONE" in refused.value.kinds

    with pytest.raises(MaskingError) as forged:
        mask_state({"output": "hi"}, after_mask=lambda s: {**s, "output": "<PERSON_9>"})
    assert "surrogate" in forged.value.kinds

    with pytest.raises(MaskingError) as keyed:
        mask_state({"output": "hi"}, after_mask=lambda s: {**s, "input": {"patient": "Asha"}})
    assert "key" in keyed.value.kinds


def test_a_cut_that_runs_after_masking_cannot_expose_a_value():
    def cut(state):
        return {**state, "output": state["output"][:20]}

    payload = mask_state({"output": "ring 9876543210 and also 9123456780 please"}, fit=cut)
    assert "98765" not in payload.text


def test_a_payload_cannot_be_built_by_hand_and_hides_its_map():
    with pytest.raises(TypeError):
        MaskedPayload(text="raw 9876543210", state_json="", mapping={})
    payload = mask_state({"output": "call 9876543210"})
    assert "9876543210" not in repr(payload)


@pytest.mark.covers("masking-fail-closed")
@pytest.mark.parametrize("marker", [None, "0" * 40])
def test_a_model_directory_that_is_not_the_pinned_revision_is_refused(settings, tmp_path, marker):
    (tmp_path / "config.json").write_text("{}")
    if marker:
        (tmp_path / ".revision").write_text(marker)
    settings.MASKING_NER_BACKEND = ""
    settings.MASKING_MODEL_DIR = str(tmp_path)
    settings.MASKING_MODEL_REVISION = "a" * 40

    with pytest.raises(MaskingError) as refused:
        masked("a sentence that is long enough to be read by the model, surely")

    assert refused.value.reason == "model_revision"


@pytest.mark.parametrize(
    ("text", "gone"),
    [
        ("order to Lake View Apartments, Salt Lake 700091", ["Lake View Apartments", "Salt Lake"]),
        ("deliver at Flat 4B, Sector 62, Noida", ["Flat 4B", "Sector 62", "Noida"]),
        ("House no. 12/3 near Gandhi Park gate", ["12/3", "Gandhi Park"]),
        ("मेरा पता लेक व्यू अपार्टमेंट, साल्ट लेक है", ["लेक व्यू अपार्टमेंट", "साल्ट लेक"]),
        ("আমার ঠিকানা গ্রিন পার্ক রেসিডেন্সি, যাদবপুর", ["রেসিডেন্সি", "যাদবপুর"]),
        ("Rose Garden Heights, Koramangala", ["Rose Garden Heights", "Koramangala"]),
        ("Tolichowki ke paas Sunrise Towers mein", ["Sunrise Towers"]),
    ],
)
def test_places_are_masked_by_suffix_unit_landmark_and_gazetteer(text, gone):
    out = masked(text)

    for fragment in gone:
        assert fragment not in out


@pytest.mark.parametrize(
    "number", ["POL/2023/94503", "ABC-12345678", "LIC/2021/00123456", "1234567/89", "FL-889900"]
)
def test_segmented_policy_and_folio_numbers_are_accounts(number):
    out = masked(f"Your reference {number} is open")

    assert number not in out
    assert "<ACCOUNT_1>" in out


@pytest.mark.parametrize(
    ("text", "reading"),
    [
        ("ECG shows 72 bpm", "72"),
        ("ferritin came back 18.5 yesterday", "18.5"),
        ("फेरिटिन में 18 आया", "18"),
        ("ইসিজি তে 72 এসেছে", "72"),
        ("TSH level is 6.2", "6.2"),
        ("sugar level is 180", "180"),
    ],
)
def test_a_reading_within_three_tokens_of_a_test_or_lab_word_is_a_value(text, reading):
    assert reading not in masked(text).replace("<VALUE_1>", "")


def test_a_date_after_a_test_is_a_date_not_a_value():
    out = masked("MRI on 12 March 2026")

    assert "<DATE_1>" in out
    assert "<VALUE" not in out


@pytest.mark.parametrize(
    "term", ["maa", "মা", "मां", "ferritin", "tamsulosin", "fibromyalgia", "ইসিজি", "फेरिटिन"]
)
def test_the_widened_lexicon_terms_are_masked(term):
    assert term not in masked(f"told by {term} today")


@pytest.mark.parametrize(
    ("text", "gone"),
    [
        ("Address: Shanti Kutir, Wakad. Call later", ["Shanti Kutir", "Wakad"]),
        ("please deliver to Ocean Pearl Mulund by Friday", ["Ocean Pearl", "Mulund"]),
        ("ঠিকানা আনন্দ বাটিকা, সোদপুর।", ["আনন্দ বাটিকা", "সোদপুর"]),
        (
            "started linagliptin for gingivitis and cellulitis",
            ["linagliptin", "gingivitis", "cellulitis"],
        ),
        ("the result is 113.9 today", ["113.9"]),
    ],
)
def test_the_cue_morphology_and_decimal_backstops_catch_words_no_list_holds(text, gone):
    out = masked(text)

    for fragment in gone:
        assert fragment not in out


def test_a_sentence_end_after_a_devanagari_or_bengali_term_still_ends_the_word():
    assert "ইসিজি" not in masked("পরীক্ষা: ইসিজি।")
    assert "फेरिटिन" not in masked("जांच फेरिटिन।")


class ContextModel:
    """Stands in for the zero-shot model: finds one fixed phrase and records the chunks it saw."""

    def __init__(self, phrase, label):
        self.phrase, self.label, self.seen = phrase, label, []

    def predict_entities(self, text, labels, threshold):
        self.seen.append((text, tuple(labels), threshold))
        at = text.find(self.phrase)
        if at < 0:
            return []
        return [{"start": at, "end": at + len(self.phrase), "label": self.label, "score": 0.9}]


def test_the_context_detector_maps_labels_to_surrogate_types_and_offsets_each_chunk():
    from overbae.services.masking import gliner_ner

    text = " ".join(["word"] * 160) + " the nerve conduction study went well"
    model = ContextModel("nerve conduction study", "medical test")

    hits = gliner_ner.Detector(model, 0.15).detect(text)

    assert [(text[h.start : h.end], h.type) for h in hits] == [("nerve conduction study", "TEST")]
    assert len(model.seen) == 2
    assert model.seen[0][2] == 0.15
    assert {"medical condition", "bank or lender", "city"} <= set(model.seen[0][1])


def test_the_union_takes_the_hits_of_every_member():
    from overbae.services.masking import ner
    from overbae.services.masking.patterns import Hit

    class Fixed:
        def __init__(self, hit):
            self.hit = hit

        def detect(self, text):
            return [self.hit]

    union = ner.Union(Fixed(Hit(0, 2, "PERSON")), Fixed(Hit(3, 5, "TEST")))

    assert union.detect("ab cd") == [Hit(0, 2, "PERSON"), Hit(3, 5, "TEST")]


@pytest.mark.covers("masking-fail-closed")
def test_a_context_model_directory_that_is_not_the_pinned_revision_is_refused(tmp_path):
    from overbae.services.masking import gliner_ner

    (tmp_path / ".revision").write_text("0" * 40)

    with pytest.raises(MaskingError) as refused:
        gliner_ner.load(str(tmp_path), "a" * 40, 0.15)

    assert refused.value.reason == "model_revision"
