# Vendored from LatentAILabs/chitragupta-pii-indic at ad42bdadf9f4ec9884c4bdbf3c1b710ee52aa795 (Apache-2.0; see LICENSE and NOTICE).
# Unchanged except for the import of the sibling module. Reviewed: it imports only re, random, string.
"""Value generators, renderers and validators for Indian PII.

Every generator returns a canonical value that passes the matching validator. Renderers turn a canonical value into the
surface forms seen in real text: spacing variants, native-script digits, spoken-form digits. Labels are attached to the
rendered string by the caller, so span boundaries are exact.
"""
from __future__ import annotations

import random
import re
import string

ALNUM = string.digits + string.ascii_uppercase

# ---------------------------------------------------------------- checksums

_VD = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5], [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
       [3, 4, 0, 1, 2, 8, 9, 5, 6, 7], [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
       [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3], [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
       [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]
_VP = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4], [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
       [8, 9, 1, 6, 0, 4, 3, 5, 2, 7], [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
       [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]]
_VINV = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]


def verhoeff_check_digit(digits: str) -> str:
    c = 0
    for i, d in enumerate(reversed(digits)):
        c = _VD[c][_VP[(i + 1) % 8][int(d)]]
    return str(_VINV[c])


def verhoeff_valid(digits: str) -> bool:
    c = 0
    for i, d in enumerate(reversed(digits)):
        c = _VD[c][_VP[i % 8][int(d)]]
    return c == 0


def luhn_check_digit(digits: str) -> str:
    total = 0
    for i, d in enumerate(reversed(digits)):
        n = int(d)
        if i % 2 == 0:
            n *= 2
            n -= 9 if n > 9 else 0
        total += n
    return str((10 - total % 10) % 10)


def luhn_valid(digits: str) -> bool:
    return luhn_check_digit(digits[:-1]) == digits[-1]


def gstin_check_char(first14: str) -> str:
    total = 0
    for i, ch in enumerate(first14):
        v = ALNUM.index(ch) * (2 if i % 2 else 1)
        total += v // 36 + v % 36
    return ALNUM[(36 - total % 36) % 36]


# ---------------------------------------------------------------- generators (canonical values)

STATE_CODES = ["AP", "AR", "AS", "BR", "CG", "GA", "GJ", "HR", "HP", "JH", "KA", "KL", "MP", "MH", "MN", "ML", "MZ",
               "NL", "OD", "PB", "RJ", "SK", "TN", "TS", "TR", "UP", "UK", "WB", "DL", "JK", "CH", "PY"]
UPI_HANDLES = ["okaxis", "oksbi", "okhdfcbank", "okicici", "ybl", "paytm", "ibl", "axl", "upi", "apl", "jupiteraxis"]
EMAIL_DOMAINS = ["gmail.com", "yahoo.co.in", "outlook.com", "rediffmail.com", "hotmail.com", "proton.me"]
PAN_HOLDER = "PPPPPPPPCHFATBLJG"  # weighted towards individuals


def aadhaar(rng: random.Random) -> str:
    body = str(rng.randint(2, 9)) + "".join(rng.choice(string.digits) for _ in range(10))
    return body + verhoeff_check_digit(body)


def pan(rng: random.Random, surname_initial: str | None = None) -> str:
    first3 = "".join(rng.choice(string.ascii_uppercase) for _ in range(3))
    fifth = (surname_initial or rng.choice(string.ascii_uppercase)).upper()
    return first3 + rng.choice(PAN_HOLDER) + fifth + f"{rng.randint(1, 9999):04d}" + rng.choice(string.ascii_uppercase)


def gstin(rng: random.Random) -> str:
    first14 = f"{rng.randint(1, 37):02d}" + pan(rng) + rng.choice("123456789ABCDEFGHJK") + "Z"
    return first14 + gstin_check_char(first14)


def ifsc(rng: random.Random) -> str:
    bank = rng.choice(["SBIN", "HDFC", "ICIC", "UTIB", "PUNB", "BARB", "CNRB", "KKBK", "IDIB", "UBIN", "YESB"])
    return bank + "0" + "".join(rng.choice(string.digits + string.ascii_uppercase) for _ in range(6))


def card_number(rng: random.Random) -> str:
    prefix = rng.choice(["4", "51", "52", "53", "54", "55", "60", "65", "81", "82", "508"])
    body = prefix + "".join(rng.choice(string.digits) for _ in range(15 - len(prefix)))
    return body + luhn_check_digit(body)


def bank_account(rng: random.Random) -> str:
    n = rng.choice([9, 11, 12, 14, 15, 16, 18])
    return str(rng.randint(1, 9)) + "".join(rng.choice(string.digits) for _ in range(n - 1))


def phone(rng: random.Random) -> str:
    return str(rng.randint(6, 9)) + "".join(rng.choice(string.digits) for _ in range(9))


def voter_id(rng: random.Random) -> str:
    return "".join(rng.choice(string.ascii_uppercase) for _ in range(3)) + f"{rng.randint(0, 9999999):07d}"


def passport(rng: random.Random) -> str:
    return rng.choice("ABCDEFGHJKLMNPRSTUVWY") + str(rng.randint(1, 9)) + f"{rng.randint(0, 999999):06d}"


def driving_licence(rng: random.Random) -> str:
    return rng.choice(STATE_CODES) + f"{rng.randint(1, 99):02d}" + str(rng.randint(1990, 2025)) + \
        f"{rng.randint(0, 9999999):07d}"


def vehicle_reg(rng: random.Random) -> str:
    if rng.random() < 0.1:  # Bharat series: 23 BH 1234 AB
        return f"{rng.randint(21, 25)}BH{rng.randint(1, 9999):04d}" + "".join(rng.choices(string.ascii_uppercase, k=2))
    series = "".join(rng.choices(string.ascii_uppercase, k=rng.choice([1, 2])))
    return rng.choice(STATE_CODES) + f"{rng.randint(1, 99):02d}" + series + f"{rng.randint(1, 9999):04d}"


MONTHS_EN = "January February March April May June July August September October November December".split()


def dob(rng: random.Random) -> str:
    """Canonical ISO date of birth, 1950-2008."""
    y, m = rng.randint(1950, 2008), rng.randint(1, 12)
    d = rng.randint(1, 28 if m == 2 else 30 if m in (4, 6, 9, 11) else 31)
    return f"{y:04d}-{m:02d}-{d:02d}"


def pincode(rng: random.Random) -> str:
    return str(rng.randint(1, 8)) + f"{rng.randint(10000, 99999):05d}"


def ip_address(rng: random.Random) -> str:
    return ".".join(str(rng.randint(1, 254)) for _ in range(4))


def upi_id(rng: random.Random, name_ascii: str) -> str:
    base = re.sub(r"[^a-z0-9.]", "", name_ascii.lower().replace(" ", rng.choice([".", "", ""])))[:20] or "user"
    if rng.random() < 0.4:
        base = phone(rng)
    elif rng.random() < 0.4:
        base += str(rng.randint(1, 999))
    return f"{base}@{rng.choice(UPI_HANDLES)}"


def email(rng: random.Random, name_ascii: str) -> str:
    parts = name_ascii.lower().split() or ["user"]
    base = rng.choice([".".join(parts), "".join(parts), parts[0] + str(rng.randint(1, 9999)), parts[0][0] + parts[-1]])
    return re.sub(r"[^a-z0-9.]", "", base) + "@" + rng.choice(EMAIL_DOMAINS)


# ---------------------------------------------------------------- validators (for the post-processing layer)

VALIDATORS = {
    "AADHAAR": lambda v: bool(re.fullmatch(r"[2-9]\d{11}", v)) and verhoeff_valid(v),
    "PAN": lambda v: bool(re.fullmatch(r"[A-Z]{3}[PCHFATBLJG][A-Z]\d{4}[A-Z]", v)),
    "GSTIN": lambda v: bool(re.fullmatch(r"\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]", v)) and gstin_check_char(v[:14]) == v[14],
    "IFSC": lambda v: bool(re.fullmatch(r"[A-Z]{4}0[A-Z0-9]{6}", v)),
    "CARD_NUMBER": lambda v: bool(re.fullmatch(r"\d{13,19}", v)) and luhn_valid(v),
    "PHONE": lambda v: bool(re.fullmatch(r"[6-9]\d{9}", v)),
    "VOTER_ID": lambda v: bool(re.fullmatch(r"[A-Z]{3}\d{7}", v)),
    "PASSPORT": lambda v: bool(re.fullmatch(r"[A-Z][1-9]\d{6}", v)),
    "PINCODE": lambda v: bool(re.fullmatch(r"[1-8]\d{5}", v)),
    "VEHICLE_REG": lambda v: bool(re.fullmatch(r"[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}|\d{2}BH\d{4}[A-Z]{2}", v)),
}

# ---------------------------------------------------------------- renderers (surface forms)

DIGIT_ZERO = {"latin": 0x30, "devanagari": 0x966, "bengali": 0x9E6, "gurmukhi": 0xA66, "gujarati": 0xAE6,
              "odia": 0xB66, "tamil": 0xBE6, "telugu": 0xC66, "kannada": 0xCE6, "malayalam": 0xD66, "urdu": 0x6F0}

SPOKEN = {
    "en": "zero one two three four five six seven eight nine".split(),
    "hi_latn": "shunya ek do teen chaar paanch chhe saat aath nau".split(),
    "hi_deva": "शून्य एक दो तीन चार पाँच छह सात आठ नौ".split(),
}


def to_script_digits(s: str, script: str) -> str:
    z = DIGIT_ZERO[script]
    return "".join(chr(z + int(c)) if c.isascii() and c.isdigit() else c for c in s)


def normalise_digits(s: str) -> str:
    """Map any native-script digit back to ASCII (used before validation)."""
    out = []
    for c in s:
        for z in DIGIT_ZERO.values():
            if z <= ord(c) <= z + 9:
                c = str(ord(c) - z)
                break
        out.append(c)
    return "".join(out)


def to_spoken(digits: str, lang: str, rng: random.Random) -> str:
    words = SPOKEN[lang]
    out, i = [], 0
    while i < len(digits):
        if lang == "en" and i + 1 < len(digits) and digits[i] == digits[i + 1] and rng.random() < 0.5:
            out.append("double " + words[int(digits[i])])
            i += 2
            continue
        out.append(words[int(digits[i])])
        i += 1
    return " ".join(out)


def group(s: str, sizes: list[int], sep: str) -> str:
    parts, i = [], 0
    for n in sizes:
        parts.append(s[i:i + n])
        i += n
    if i < len(s):
        parts.append(s[i:])
    return sep.join(p for p in parts if p)


def render(label: str, value: str, rng: random.Random, script: str = "latin", spoken_lang: str | None = None) -> str:
    """One realistic surface form of a canonical value."""
    if spoken_lang and label in {"PHONE", "AADHAAR", "PINCODE", "BANK_ACCOUNT"}:
        return to_spoken(value, spoken_lang, rng)
    if label == "PHONE":
        v = rng.choice([value, group(value, [5, 5], " "), group(value, [3, 3, 4], "-"), group(value, [5, 5], "-")])
        v = rng.choice(["", "", "+91 ", "+91-", "0", "91"]) + v
    elif label == "AADHAAR":
        v = rng.choice([group(value, [4, 4, 4], " "), group(value, [4, 4, 4], " "), value, group(value, [4, 4, 4], "-")])
    elif label == "CARD_NUMBER":
        v = rng.choice([group(value, [4, 4, 4, 4], " "), value, group(value, [4, 4, 4, 4], "-")])
    elif label == "VEHICLE_REG":
        m = re.fullmatch(r"([A-Z]{2})(\d{2})([A-Z]{1,2})(\d{4})", value)
        v = rng.choice([value, " ".join(m.groups()), "-".join(m.groups())]) if m else value
    elif label == "DRIVING_LICENCE":
        v = rng.choice([value, value[:4] + " " + value[4:], value[:2] + "-" + value[2:4] + "-" + value[4:]])
    elif label == "DOB":
        y, m, d = (int(x) for x in value.split("-"))
        v = rng.choice([f"{d:02d}/{m:02d}/{y}", f"{d:02d}/{m:02d}/{y}", f"{d}/{m}/{y}", f"{d:02d}-{m:02d}-{y}",
                        f"{d:02d}.{m:02d}.{y}", f"{d} {MONTHS_EN[m - 1]} {y}", f"{d} {MONTHS_EN[m - 1][:3]} {y}",
                        f"{MONTHS_EN[m - 1]} {d}, {y}"])
    elif label == "PINCODE":
        v = rng.choice([value, value, value[:3] + " " + value[3:]])
    elif label in {"PAN", "IFSC", "GSTIN", "VOTER_ID", "PASSPORT"}:
        v = value if rng.random() < 0.9 else value.lower()
    else:
        v = value
    if label in {"AADHAAR", "CARD_NUMBER", "PHONE", "BANK_ACCOUNT"} and rng.random() < 0.08:
        digits = [i for i, c in enumerate(v) if c.isdigit()]
        keep = set(digits[-4:]) | (set(digits[:2]) if label == "PHONE" else set())
        mask = rng.choice("XX*x")
        v = "".join(mask if (i in digits and i not in keep) else c for i, c in enumerate(v))
    if script != "latin" and label in {"PHONE", "AADHAAR", "PINCODE", "BANK_ACCOUNT", "CARD_NUMBER", "DOB"}:
        v = to_script_digits(v, script)
    return v


def canonical(label: str, surface: str) -> str:
    """Strip formatting from a detected span so it can be validated."""
    s = normalise_digits(surface).upper()
    if label == "PHONE":
        s = re.sub(r"\D", "", s)
        s = s[-10:] if len(s) >= 10 else s
    elif label in {"AADHAAR", "CARD_NUMBER", "PINCODE", "BANK_ACCOUNT"}:
        s = re.sub(r"\D", "", s)
    else:
        s = re.sub(r"[\s\-./]", "", s)
    return s


GENERATORS = {"AADHAAR": aadhaar, "PAN": pan, "GSTIN": gstin, "IFSC": ifsc, "CARD_NUMBER": card_number,
              "BANK_ACCOUNT": bank_account, "PHONE": phone, "VOTER_ID": voter_id, "PASSPORT": passport,
              "DRIVING_LICENCE": driving_licence, "VEHICLE_REG": vehicle_reg, "PINCODE": pincode,
              "IP_ADDRESS": ip_address, "DOB": dob}


# ---------------------------------------------------------------- test-only surface forms
# Formats here never come out of render(); the test set uses them so ID scores measure generalisation, not
# memorised formatting. test_ids.py checks the two sets of shapes are disjoint.

TEST_UPI_HANDLES = ["fbl", "airtel", "kotak", "sbi", "icici", "hdfcbank", "abfspay", "waaxis", "freecharge"]
TEST_EMAIL_DOMAINS = ["zohomail.in", "icloud.com", "live.in", "gmx.com", "yandex.com", "mail.com"]


def card_number_test(rng: random.Random) -> str:
    """Amex (15) or Diners (14): lengths render()/card_number() never produce."""
    if rng.random() < 0.5:
        body = rng.choice(["34", "37"]) + "".join(rng.choice(string.digits) for _ in range(12))
    else:
        body = rng.choice(["300", "305", "36", "38"])
        body += "".join(rng.choice(string.digits) for _ in range(13 - len(body)))
    return body + luhn_check_digit(body)


def ipv6(rng: random.Random) -> str:
    return ":".join(f"{rng.randint(0, 0xffff):x}" for _ in range(8))


def render_test(label: str, value: str, rng: random.Random, script: str = "latin", spoken_lang: str | None = None) -> str:
    if spoken_lang and label in {"PHONE", "AADHAAR", "PINCODE", "BANK_ACCOUNT"}:
        words = SPOKEN[spoken_lang]
        groups = [value[i:i + 2] for i in range(0, len(value), 2)]  # spoken in pairs with commas
        return ", ".join(" ".join(words[int(c)] for c in g) for g in groups)
    v = value
    if label == "PHONE":
        v = rng.choice([f"(+91) {group(value, [5, 5], ' ')}", "+91 " + group(value, [2, 2, 2, 2, 2], " "),
                        "91-" + group(value, [5, 5], "-"), "0091-" + value, group(value, [5, 5], "."),
                        "+91 " + group(value, [4, 3, 3], " ")])
    elif label == "AADHAAR":
        v = rng.choice([group(value, [4, 4, 4], "."), group(value, [4, 4, 4], "  "), group(value, [4, 4, 4], "/")])
        if rng.random() < 0.15:
            v = rng.choice(["•••• •••• ", "••••-••••-"]) + value[-4:]
    elif label == "PAN":
        v = rng.choice([value[:5] + " " + value[5:9] + " " + value[9], value[:5] + "-" + value[5:9] + "-" + value[9]])
    elif label == "CARD_NUMBER":
        if len(value) == 15:
            v = group(value, [4, 6, 5], rng.choice([" ", "-"]))
        elif len(value) == 14:
            v = group(value, [4, 6, 4], rng.choice([" ", "-"]))
        else:
            v = value[:4] + " **** **** " + value[-4:]
    elif label == "VEHICLE_REG":
        m = re.fullmatch(r"([A-Z]{2})(\d{2})([A-Z]{1,2})(\d{4})", value)
        v = rng.choice([".".join(m.groups()), m.group(1) + " " + m.group(2) + " " + m.group(3) + "-" + m.group(4),
                        value.lower()]) if m else value[:2] + " " + value[2:4] + " " + value[4:8] + " " + value[8:]
    elif label == "DRIVING_LICENCE":
        v = value[:2] + "/" + value[2:4] + "/" + value[4:8] + "/" + value[8:]
    elif label == "IFSC":
        v = value[:4] + " " + value[4:]
    elif label == "PINCODE":
        v = value[:3] + "-" + value[3:]
    elif label == "BANK_ACCOUNT":
        v = group(value, [4, 4, 4, 4, 4], rng.choice([" ", "-"]))
    elif label == "VOTER_ID":
        v = value[:3] + rng.choice([" ", "/"]) + value[3:]
    elif label == "PASSPORT":
        v = value[0] + " " + value[1:]
    elif label == "GSTIN":
        v = value[:2] + " " + value[2:12] + " " + value[12:]
    elif label == "DOB":
        y, m, d = (int(x) for x in value.split("-"))
        suf = "th" if 11 <= d <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(d % 10, "th")
        v = rng.choice([f"{d}{suf} {MONTHS_EN[m - 1]} {y}", value, f"{d:02d}-{MONTHS_EN[m - 1][:3]}-{str(y)[2:]}",
                        f"{MONTHS_EN[m - 1]} {d}{suf}, {y}", f"{d:02d}/{m:02d}/{str(y)[2:]}"])
    if script != "latin" and label in {"PHONE", "AADHAAR", "PINCODE", "BANK_ACCOUNT", "CARD_NUMBER", "DOB"}:
        v = to_script_digits(v, script)
    return v


def upi_id_test(rng: random.Random, name_ascii: str) -> str:
    return upi_id(rng, name_ascii).split("@")[0] + "@" + rng.choice(TEST_UPI_HANDLES)


def email_test(rng: random.Random, name_ascii: str) -> str:
    return email(rng, name_ascii).split("@")[0] + "@" + rng.choice(TEST_EMAIL_DOMAINS)


def shape(s: str) -> str:
    """Format signature: digits -> 9, letters -> A (native digits normalised first)."""
    return re.sub(r"[a-z]", "a", re.sub(r"[A-Z]", "A", re.sub(r"\d", "9", normalise_digits(s))))


# ---------------------------------------------------------------- training-time format augmentation
# Random groupings and separators so the tagger learns that formatting varies, without ever producing a shape
# render_test() can produce (the unseen-format test slice stays unseen; test_ids.py checks this).

# whole separator / prefix families are kept out of training (render() never uses them on IDs either), so test
# spans written with them stay unseen in spirit, not just in exact shape; eval reports them as "unseen_strict"
HELD_OUT = re.compile(r"[./•]|  |\(\+91\)|^0091")
AUG_SEPS = [" ", "-", " - ", ""]
AUG_PHONE_PREFIX = ["", "+91", "91", "0", "+91-", "(0)"]
AUG_LABELS = {"PHONE", "AADHAAR", "CARD_NUMBER", "BANK_ACCOUNT", "PINCODE", "PAN", "GSTIN", "IFSC", "VOTER_ID",
              "PASSPORT", "DRIVING_LICENCE", "VEHICLE_REG"}
_TEST_SHAPES: dict = {}


def _test_shapes(label: str) -> set:
    if label not in _TEST_SHAPES:
        rng = random.Random(12345)
        gen = GENERATORS[label]
        vals = [card_number_test(rng) if label == "CARD_NUMBER" and i % 5 else gen(rng) for i in range(4000)]
        _TEST_SHAPES[label] = {shape(render_test(label, v, rng)) for v in vals}
    return _TEST_SHAPES[label]


def _random_groups(s: str, rng: random.Random) -> list[str]:
    parts, i = [], 0
    while i < len(s):
        n = rng.randint(2, 6)
        parts.append(s[i:i + n]); i += n
    return parts


def render_aug(label: str, value: str, rng: random.Random, script: str = "latin") -> str | None:
    """A random surface form of `value` whose shape is not a test-only shape; None if none found."""
    for _ in range(20):
        sep = rng.choice(AUG_SEPS)
        if label in {"PHONE", "AADHAAR", "CARD_NUMBER", "BANK_ACCOUNT", "PINCODE"}:
            v = sep.join(_random_groups(value, rng)) if rng.random() < 0.85 else value
            if label == "PHONE":
                pre = rng.choice(AUG_PHONE_PREFIX)
                v = (pre + rng.choice([" ", "-", ""]) if pre else "") + v
        else:  # alphanumeric IDs: people break them where letters meet digits, so split only there
            runs = re.findall(r"[A-Z]+|\d+", value)
            cut = sorted(rng.sample(range(1, len(runs)), k=rng.randint(1, len(runs) - 1))) if len(runs) > 1 else []
            v = sep.join("".join(runs[i:j]) for i, j in zip([0] + cut, cut + [len(runs)]))
            if rng.random() < 0.15:
                v = v.lower()
        if sep == "" and v == value:
            continue
        if script != "latin" and label in {"PHONE", "AADHAAR", "PINCODE", "BANK_ACCOUNT", "CARD_NUMBER"}:
            v = to_script_digits(v, script)
        if shape(v) not in _test_shapes(label) and not HELD_OUT.search(normalise_digits(v)):
            return v
    return None
