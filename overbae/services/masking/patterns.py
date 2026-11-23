"""Regex detectors: identifiers (with checksums), amounts, dates, measured values.

Each detector returns ``Hit`` spans over the text it was given. Overlaps are resolved by the
masker, not here. ``\\d`` matches digits in every script, so Devanagari and Bengali numerals
are covered as well.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass

from overbae.services.masking.checksums import aadhaar_valid, card_valid

# Higher wins an overlap of equal length.
PRIORITY = {
    "EMAIL": 90,
    "URL": 85,
    "ACCOUNT": 80,
    "PHONE": 75,
    "ID": 70,
    "AMOUNT": 72,
    "DATE": 55,
    "VALUE": 50,
    "PERSON": 45,
    "ADDRESS": 44,
    "LOCATION": 43,
    "ORG": 42,
    "MEDICATION": 40,
    "CONDITION": 39,
    "TEST": 38,
    "RELATION": 30,
}


@dataclass(frozen=True)
class Hit:
    start: int
    end: int
    type: str

    @property
    def priority(self) -> int:
        return PRIORITY.get(self.type, 0)


# The danda (U+0964, U+0965) ends a sentence; it is not part of a word.
WORD = r"[\wऀ-ॣ०-ॿঀ-৿]"
NOT_WORD_BEFORE = rf"(?<!{WORD})"
NOT_WORD_AFTER = rf"(?!{WORD})"

_FLAGS = re.IGNORECASE

EMAIL = re.compile(r"[\w.%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
# A UPI address: handle@bank with no dot after the @.
UPI = re.compile(rf"{NOT_WORD_BEFORE}[\w.\-]{{2,}}@[A-Za-z][A-Za-z0-9]{{2,}}{NOT_WORD_AFTER}(?!\.)")
URL = re.compile(r"(?:https?://|www\.)[^\s<>\"')\]]+", _FLAGS)
IPV4 = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
AADHAAR = re.compile(r"(?<!\d)\d{4}[ \-]?\d{4}[ \-]?\d{4}(?!\d)")
PAN = re.compile(r"(?<![A-Za-z0-9])[A-Z]{5}\d{4}[A-Z](?![A-Za-z0-9])")
IFSC = re.compile(r"(?<![A-Za-z0-9])[A-Z]{4}0[A-Z0-9]{6}(?![A-Za-z0-9])")
GSTIN = re.compile(r"(?<![A-Za-z0-9])\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z](?![A-Za-z0-9])")
PHONE = re.compile(
    r"(?<![\w])(?:\+?\s?91[\s\-]?|0)?[6-9]\d{4}[\s\-]?\d{5}(?!\d)"
    r"|(?<![\w])\+\d{1,3}[\s\-]?\(?\d{2,5}\)?(?:[\s\-]?\d{2,5}){1,3}(?!\d)"
)
CARD = re.compile(r"(?<!\d)\d(?:[ \-]?\d){12,18}(?!\d)")
# An alphanumeric identifier: letters then 5 or more digits (voter id, passport, licence, policy).
ALNUM_ID = re.compile(r"(?<![A-Za-z0-9])[A-Z]{1,5}[\- ]?\d{5,}[A-Z0-9]*(?![A-Za-z0-9])")
# Two or more letters, then a separator and five or more digits: POL/2023/94503, ABC-12345678.
SEGMENTED_ID = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]{2,6}(?:[/\-]\d{1,6})*[/\-]\d{5,}(?![\d/])")
# A folio: five or more digits, a slash, then a short number (1234567/89).
FOLIO = re.compile(r"(?<![\d/.])\d{5,}/\d{1,3}(?![\d/])")
DIGIT_RUN = re.compile(r"\d{6,}")
SPACED_DIGITS = re.compile(r"(?<!\d)\d(?:[ \-.]?\d){8,}(?!\d)")

_ACCOUNT_KEYWORD = (
    r"(?:a/?c|acct|account|policy|folio|loan|lan|emi|customer|cust|card|upi|vpa|"
    r"खाता|खाते|पॉलिसी|অ্যাকাউন্ট|একাউন্ট|পলিসি|হিসাব)"
)
ACCOUNT_NUMBER = re.compile(
    rf"{_ACCOUNT_KEYWORD}\W{{0,12}}(?:no\.?|number|num|id|#|संख्या|नंबर|নম্বর)?\W{{0,6}}"
    r"((?=[A-Z0-9/\-]*\d)[A-Z0-9][A-Z0-9/\-]{5,})",
    _FLAGS,
)

# --- amounts -------------------------------------------------------------------------------

_NUMBER = r"(?:\d{1,3}(?:,\d{2,3})+|\d+)(?:\.\d+)?"
_SCALE = (
    r"lakhs?|lacs?|crores?|cr|thousand|million|billion|mn|bn|k|m|"
    r"लाख|करोड़|करोड|हज़ार|हजार|লাখ|কোটি|হাজার"
)
_CURRENCY_WORD = (
    r"rupees?|rs|inr|dollars?|usd|euros?|eur|pounds?|gbp|taka|bdt|"
    r"रुपये|रुपए|रुपया|रु|টাকা|রুপি|টাকার"
)
AMOUNT_PREFIX = re.compile(
    rf"(?:₹|Rs\.?|INR|\$|USD|€|EUR|£|GBP|৳|BDT)\s?{_NUMBER}(?:\s?(?:{_SCALE}){NOT_WORD_AFTER})?",
    _FLAGS,
)
AMOUNT_SUFFIX = re.compile(
    rf"{_NUMBER}\s?(?:(?:{_SCALE})\s?)?(?:{_CURRENCY_WORD}){NOT_WORD_AFTER}"
    rf"|{_NUMBER}\s?(?:{_SCALE}){NOT_WORD_AFTER}",
    _FLAGS,
)
_MONEY_WORD = (
    r"emi|salary|balance|bill|price|fee|fees|rent|loan|premium|deposit|payment|paid|charge|"
    r"charges|cost|amount|interest|tax|due|outstanding|installment|instalment|refund|"
    r"वेतन|तनख्वाह|किस्त|बिल|किराया|शुल्क|ब्याज|कर्ज|बकाया|भुगतान|"
    r"বেতন|কিস্তি|বিল|ভাড়া|ফি|সুদ|ঋণ|বকেয়া|পরিশোধ"
)
AMOUNT_NEAR_WORD = re.compile(
    rf"(?:{_MONEY_WORD})\W{{0,15}}(?:of|is|was|rs|inr)?\W{{0,6}}({_NUMBER})",
    _FLAGS,
)
AMOUNT_BEFORE_WORD = re.compile(rf"({_NUMBER})\s?\W?\s?(?:{_MONEY_WORD}){NOT_WORD_AFTER}", _FLAGS)

# --- dates and times -----------------------------------------------------------------------

_MONTH = (
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|"
    r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?|"
    r"जनवरी|फरवरी|फ़रवरी|मार्च|अप्रैल|मई|जून|जुलाई|अगस्त|सितंबर|सितम्बर|अक्टूबर|नवंबर|नवम्बर|दिसंबर|दिसम्बर|"
    r"জানুয়ারি|ফেব্রুয়ারি|মার্চ|এপ্রিল|মে|জুন|জুলাই|আগস্ট|সেপ্টেম্বর|অক্টোবর|নভেম্বর|ডিসেম্বর"
)
_TIME = (
    r"\d{1,2}:\d{2}(?::\d{2})?(?:\s?(?:am|pm))?|\d{1,2}[.:]\d{2}\s?(?:am|pm)|\d{1,2}\s?(?:am|pm)"
)
DATE_ISO = re.compile(
    r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?(?:Z|[+\-]\d{2}:?\d{2})?)?"
)
DATE_NUMERIC = re.compile(r"(?<![\d/])\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}(?![\d/])")
DATE_DAY_MONTH = re.compile(
    rf"\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:{_MONTH}){NOT_WORD_AFTER}\.?(?:,?\s+\d{{2,4}})?",
    _FLAGS,
)
DATE_MONTH_DAY = re.compile(
    rf"(?<![\w])(?:{_MONTH}){NOT_WORD_AFTER}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{2,4}})?",
    _FLAGS,
)
DATE_ORDINAL = re.compile(r"(?<![\w])\d{1,2}(?:st|nd|rd|th)(?![\w])", _FLAGS)
DATE_TARIKH = re.compile(r"\d{1,2}\s*(?:तारीख|तारिख|তারিখ)|(?:तारीख|तारिख|তারিখ)\s*\d{1,2}")
TIME_OF_DAY = re.compile(rf"(?<![\d:.]){_TIME}(?![\d:])", _FLAGS)

# --- measured values -----------------------------------------------------------------------

_UNIT = (
    r"mg/dl|mg/l|mmol/l|mmol|mmhg|mcg|µg|ug|mg|kg|kcal|bpm|iu|units?|ml|cm|mm|g|%|°\s?[cf]|lbs?|"
    r"মি\.গ্রা|কেজি|কিলো|मि\.ग्रा|किलो"
)
VALUE_UNIT = re.compile(rf"(?<![\w.]){_NUMBER}\s?(?:{_UNIT})(?![\w/])", _FLAGS)
_MEASURE_WORD = (
    r"bp|blood pressure|sugar|glucose|hba1c|a1c|cholesterol|dose|dosage|weight|height|bmi|pulse|"
    r"spo2|oxygen|temperature|temp|creatinine|haemoglobin|hemoglobin|hb|tsh|ldl|hdl|platelets?|"
    r"wbc|rbc|fasting|tsh|t3|t4|ferritin|vitamin d|vitamin b12|uric acid|urea|bilirubin|sgpt|sgot|"
    r"alt|ast|esr|crp|egfr|inr|psa|hematocrit|haematocrit|lipid|triglycerides?|"
    r"हीमोग्लोबिन|फेरिटिन|क्रिएटिनिन|कोलेस्ट्रॉल|यूरिक एसिड|"
    r"হিমোগ্লোবিন|ফেরিটিন|ক্রিয়েটিনিন|কোলেস্টেরল|ইউরিক অ্যাসিড|"
    r"शुगर|चीनी|ब्लड प्रेशर|बीपी|वजन|वज़न|तापमान|बुखार|खुराक|"
    r"সুগার|চিনি|প্রেশার|ওজন|তাপমাত্রা|জ্বর|ডোজ"
)
# Up to two words (no digits) may stand between the term and its number: "sugar level is 180".
_GAP = rf"(?:[\s:=\-,;(]+(?!\d){WORD}+){{0,2}}[\s:=\-,;(]*"
_NUMERIC_VALUE = r"(\d+(?:\.\d+)?(?:\s?/\s?\d+(?:\.\d+)?)?)(?![\d.]\d)"
VALUE_AFTER_WORD = re.compile(rf"(?:{_MEASURE_WORD}){_GAP}{_NUMERIC_VALUE}", _FLAGS)
VALUE_AFTER_TEST = re.compile(rf"{_GAP}{_NUMERIC_VALUE}", _FLAGS)
# The safe side: a decimal of two or more with one or two places is a reading. The platform's
# own scores (0.00 to 1.0) in a judge prompt stay clear of it.
BARE_DECIMAL = re.compile(r"(?<![\d.,/\-])(?:[2-9]|\d{2,3})\.\d{1,2}(?![\d.,/]\d|\d|\.\d)")
BLOOD_PRESSURE = re.compile(r"(?<![\d/])\d{2,3}\s?/\s?\d{2,3}(?![\d/])")


# --- addresses -----------------------------------------------------------------------------

_CAP = r"(?-i:[A-Z])[\w'’\-]*"
_PLACE_SUFFIX = (
    r"towers?|apartments?|residency|residences?|heights|enclave|society|complex|villas?|nagar|"
    r"colony|gardens?|plaza|road|rd|lane|street|st|marg|chowk|bazaar|bazar|gali|layout|"
    r"extension|park|vihar|puram|pur|abad|ganj|garh|bagh|hills|valley|estate|court|manor|"
    r"mansion|bhavan|bhawan|niwas|nivas|sadan|kunj|"
    r"अपार्टमेंट|अपार्टमेन्ट|टावर्स|टावर|रेजीडेंसी|रेजिडेंसी|हाइट्स|एनक्लेव|सोसाइटी|सोसायटी|"
    r"कॉम्प्लेक्स|विला|नगर|कॉलोनी|रोड|गली|मार्ग|लेन|स्ट्रीट|विहार|पुरम|बाग|भवन|निवास|कुंज|"
    r"অ্যাপার্টমেন্ট|অ্যাপার্টমেণ্ট|টাওয়ার্স|টাওয়ার|রেসিডেন্সি|হাইটস|এনক্লেভ|সোসাইটি|"
    r"কমপ্লেক্স|ভিলা|নগর|কলোনি|রোড|গলি|লেন|স্ট্রিট|সরণি|পাড়া|বাগান|ভবন|নিবাস|কুঞ্জ"
)
_PLACE_LATIN_ALT = "|".join(w for w in _PLACE_SUFFIX.split("|") if w.isascii())
_PLACE_INDIC_ALT = "|".join(w for w in _PLACE_SUFFIX.split("|") if not w.isascii())
# One to three capitalised words (or words of another script) ending in a place suffix.
PLACE_LATIN = re.compile(
    rf"(?<![\w]){_CAP}(?:\s+{_CAP}){{0,2}}\s+(?:{_PLACE_LATIN_ALT})(?![\w])", _FLAGS
)
PLACE_INDIC = re.compile(
    rf"(?<!{WORD})(?:[ऀ-ॣ०-ॿঀ-৿]+\s+){{1,3}}(?:{_PLACE_INDIC_ALT}){NOT_WORD_AFTER}"
)
_UNIT_WORD = (
    r"flat|house|plot|door|room|unit|apt|"
    r"फ्लैट|मकान|प्लॉट|कमरा|ফ্ল্যাট|বাড়ি|প্লট|রুম"
)
UNIT_NUMBER = re.compile(
    rf"(?<!{WORD})(?:{_UNIT_WORD})\s*(?:no\.?|number|#|नं\.?|नंबर|নং|নম্বর)?\s*[A-Za-z]?\d[\w/\-]*",
    _FLAGS,
)
_AREA_WORD = r"sector|block|phase|ward|wing|tower|सेक्टर|ब्लॉक|फेज|वार्ड|সেক্টর|ব্লক|ফেজ|ওয়ার্ড"
AREA_CODE = re.compile(
    rf"(?<!{WORD})(?:{_AREA_WORD})\s*-?\s*(?:\d+[A-Za-z]?|(?-i:[A-Z])(?![A-Za-z]))(?!{WORD})",
    _FLAGS,
)
_LANDMARK_WORD = r"near|opp\.?|opposite|behind|beside|next to|nr\.?"
LANDMARK = re.compile(rf"(?<!{WORD})(?:{_LANDMARK_WORD})\s+({_CAP}(?:\s+{_CAP}){{0,3}})", _FLAGS)
LANDMARK_AFTER = re.compile(
    r"((?:[ऀ-ॣ०-ॿঀ-৿]+\s+){1,3})(?:के पास|के सामने|के पीछे|এর পাশে|র পাশে|এর কাছে|র কাছে|এর সামনে)"
)


# An address cue and the clause after it, up to the end of the sentence: "Address: Shanti Kutir,
# Wakad", "deliver to B-12 Green Park". Whatever follows the cue in that clause is masked.
_ADDRESS_CUE = (
    r"address|addr|deliver(?:ed|y)?\s+(?:to|at)|ship(?:ped)?\s+to|posted\s+to|"
    r"reside[sd]?\s+(?:at|in)|staying\s+(?:at|in)|lives?\s+(?:at|in)|"
    r"पता|पते|ठिकाना|ঠিকানা|thikana|pata|thikanay"
)
ADDRESS_CLAUSE = re.compile(
    rf"(?<!{WORD})(?:{_ADDRESS_CUE}){NOT_WORD_AFTER}\s*[:\-]?\s*([^.;।\n<\ue000]{{3,80}})",
    re.IGNORECASE,
)


def addresses(text: str) -> list[Hit]:
    hits: list[Hit] = []
    for pattern in (PLACE_LATIN, PLACE_INDIC, UNIT_NUMBER, AREA_CODE):
        hits += _spans(pattern, text, "ADDRESS")
    hits += _spans(LANDMARK, text, "ADDRESS", group=1)
    hits += _spans(LANDMARK_AFTER, text, "ADDRESS", group=1)
    hits += _spans(ADDRESS_CLAUSE, text, "ADDRESS", group=1)
    return hits


def _spans(pattern: re.Pattern, text: str, type_: str, group: int = 0) -> Iterator[Hit]:
    for match in pattern.finditer(text):
        start, end = match.span(group)
        if start < end:
            yield Hit(start, end, type_)


def structured(text: str) -> list[Hit]:
    """Identifiers: email, url, ip, aadhaar, pan, ifsc, gstin, phone, cards, accounts."""
    hits: list[Hit] = []
    hits += _spans(EMAIL, text, "EMAIL")
    hits += _spans(UPI, text, "ACCOUNT")
    hits += _spans(URL, text, "URL")
    hits += _spans(IPV4, text, "ID")
    hits += (Hit(*m.span(), "ID") for m in AADHAAR.finditer(text) if aadhaar_valid(m.group()))
    hits += _spans(PAN, text, "ID")
    hits += _spans(IFSC, text, "ACCOUNT")
    hits += _spans(GSTIN, text, "ID")
    hits += _spans(PHONE, text, "PHONE")
    hits += (Hit(*m.span(), "ACCOUNT") for m in CARD.finditer(text) if card_valid(m.group()))
    hits += _spans(ACCOUNT_NUMBER, text, "ACCOUNT", group=1)
    hits += _spans(ALNUM_ID, text, "ID")
    hits += _spans(SEGMENTED_ID, text, "ACCOUNT")
    hits += _spans(FOLIO, text, "ACCOUNT")
    # The rule that does not trust a checksum: any run of six or more digits is an identifier.
    hits += _spans(DIGIT_RUN, text, "ID")
    hits += _spans(SPACED_DIGITS, text, "ID")
    return hits


def amounts(text: str) -> list[Hit]:
    hits: list[Hit] = []
    hits += _spans(AMOUNT_PREFIX, text, "AMOUNT")
    hits += _spans(AMOUNT_SUFFIX, text, "AMOUNT")
    hits += _spans(AMOUNT_NEAR_WORD, text, "AMOUNT", group=1)
    hits += _spans(AMOUNT_BEFORE_WORD, text, "AMOUNT", group=1)
    return hits


def dates(text: str) -> list[Hit]:
    hits: list[Hit] = []
    for pattern in (
        DATE_ISO,
        DATE_NUMERIC,
        DATE_DAY_MONTH,
        DATE_MONTH_DAY,
        DATE_ORDINAL,
        DATE_TARIKH,
        TIME_OF_DAY,
    ):
        hits += _spans(pattern, text, "DATE")
    return hits


def values(text: str) -> list[Hit]:
    hits: list[Hit] = []
    hits += _spans(VALUE_UNIT, text, "VALUE")
    hits += _spans(VALUE_AFTER_WORD, text, "VALUE", group=1)
    hits += _spans(BLOOD_PRESSURE, text, "VALUE")
    hits += _spans(BARE_DECIMAL, text, "VALUE")
    return hits


# Medical morphology: a Latin word of six or more letters that ends the way a drug or a
# condition does. Over-masking a rare ordinary word is the safe error.
DRUG_WORD = re.compile(
    rf"{NOT_WORD_BEFORE}[a-z]{{3,}}(?:pril|sartan|statin|gliptin|gliflozin|olol|azole|mycin|"
    rf"cillin|floxacin|dipine|prazole|tidine|triptyline|oxetine|zepam|zolam|thiazide|semide|"
    rf"lukast|terol|navir|cycline|oxacin|parin|xaban|tinib|zumab|ximab|sartan|caine|fenac|"
    rf"profen|barbital|setron|tropium|glinide|formin|glitazone|azepine|pramine){NOT_WORD_AFTER}",
    _FLAGS,
)
CONDITION_WORD = re.compile(
    rf"{NOT_WORD_BEFORE}[a-z]{{3,}}(?:itis|emia|aemia|osis|pathy|algia|plegia|oma|trophy|"
    rf"penia|uria|ectomy|plasty|scopy|cele|sclerosis){NOT_WORD_AFTER}",
    _FLAGS,
)


def medical_words(text: str) -> list[Hit]:
    hits = list(_spans(DRUG_WORD, text, "MEDICATION"))
    hits += _spans(CONDITION_WORD, text, "CONDITION")
    return hits


_SURROGATE = re.compile(r"<[A-Z]+(?:_[A-Z]+)*_\d+>|<ESC:[A-Z_0-9]+>|<ATTACHMENT>")


def neutral(text: str) -> str:
    """``text`` with each surrogate overwritten by private-use characters of the same length, so
    the digit in ``<ID_1>`` is never read as a number and no clause runs across a surrogate.
    Offsets stay valid."""
    return _SURROGATE.sub(lambda m: "\ue000" * len(m.group()), text)


def detect(text: str) -> list[Hit]:
    text = neutral(text)
    return (
        structured(text)
        + amounts(text)
        + dates(text)
        + values(text)
        + addresses(text)
        + medical_words(text)
    )
