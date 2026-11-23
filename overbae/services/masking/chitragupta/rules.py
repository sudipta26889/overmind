# Vendored from LatentAILabs/chitragupta-pii-indic at ad42bdadf9f4ec9884c4bdbf3c1b710ee52aa795 (Apache-2.0; see LICENSE and NOTICE).
# Unchanged except for the import of the sibling module. Reviewed: it imports only re, random, string.
"""Rule layer on top of the tagger: glue ID fragments back together, then let checksums pick the label.

The tagger splits ID formats it has not seen into pieces with mixed labels ('30 JUVHQ3959X 1ZD' -> GSTIN, PAN,
GSTIN). Every structured label except BANK_ACCOUNT and DRIVING_LICENCE has a format or checksum, so a merged span can be relabelled
deterministically. Spans are (label, start, end) char offsets.
"""
import re

from overbae.services.masking.chitragupta import ids

FREE = {"PERSON", "ADDRESS"}  # free text: never merged or relabelled
SEP = re.compile(r"[\s\-./:•*]{0,3}")
MASK = re.compile(r"[•*xX]{3,}")
IPV4 = re.compile(r"(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})")
IPV6 = re.compile(r"[0-9a-fA-F]{1,4}(:[0-9a-fA-F]{0,4}){3,7}")  # >= 3 colons, so 10:30:45 is a time
# checksum labels first: a pass there is strong evidence, format-only labels are weaker
ORDER = ["GSTIN", "CARD_NUMBER", "AADHAAR", "PAN", "IFSC", "VEHICLE_REG", "VOTER_ID", "PASSPORT", "PHONE"]
# only labels a check can confirm may be switched; BANK_ACCOUNT / DRIVING_LICENCE have none, and ~10% of any
# random digit string passes Luhn or Verhoeff, so overriding them would be a coin toss
SWITCHABLE = set(ORDER) | {"IP_ADDRESS", "PHONE", "PINCODE"}
PHONE = re.compile(r"(?:0091|91|0)?[6-9]\d{9}")
DIGITS_ONLY = re.compile(r"[\d\s\-./]+")


def _word_char(c):
    return c.isalnum() or c in "•*"


def _ip(s):
    m = IPV4.fullmatch(s)
    return bool(m and all(int(g) <= 255 for g in m.groups())) or bool(IPV6.fullmatch(s))


def _valid(label, surface):
    if label == "IP_ADDRESS":
        return _ip(surface)
    if label == "PHONE":
        return bool(PHONE.fullmatch(re.sub(r"\D", "", ids.normalise_digits(surface))))
    v = ids.VALIDATORS.get(label)
    if not v:
        return False
    if label in {"CARD_NUMBER", "AADHAAR"}:
        s = ids.normalise_digits(surface)
        return bool(DIGITS_ONLY.fullmatch(s)) and v(re.sub(r"\D", "", s))
    return v(re.sub(r"[\s\-./]", "", ids.normalise_digits(surface).upper()))


def relabel(label, surface):
    if MASK.search(surface) or label not in SWITCHABLE:
        return label  # masked values cannot be validated; trust the tagger
    if _valid(label, surface):
        return label
    if _ip(surface):
        return "IP_ADDRESS"
    passing = [l for l in ORDER if _valid(l, surface)]
    if len(passing) == 1:
        return passing[0]
    digits = re.sub(r"\D", "", ids.normalise_digits(surface))
    if label in {"CARD_NUMBER", "AADHAAR"} and digits == ids.normalise_digits(re.sub(r"[\s\-./]", "", surface)) \
            and 9 <= len(digits) <= 18:
        return "BANK_ACCOUNT"  # an all-digit number that fails Luhn/Verhoeff
    return label


CHECKSUM = ["GSTIN", "CARD_NUMBER", "AADHAAR"]
GROUP = re.compile(r"[0-9A-Za-z\u0966-\u0d6f]+")


def extend(text, a, b, lo, hi):
    """The tagger sometimes stops one digit group short ('20 YZXAX4144X' of '20 YZXAX4144X DZV'). Add up to one
    neighbouring group on each side, but only when the result passes a checksum the current span fails."""
    if MASK.search(text[a:b]) or any(_valid(l, text[a:b]) for l in CHECKSUM):
        return a, b
    left = [m for m in GROUP.finditer(text, lo, a) if SEP.fullmatch(text[m.end():a])]
    right = GROUP.match(text, b + len(SEP.match(text, b).group(0))) if b < hi else None
    cands = []
    if right and right.end() <= hi: cands.append((a, right.end()))
    if left: cands.append((left[-1].start(), b))
    if left and right and right.end() <= hi: cands.append((left[-1].start(), right.end()))
    for na, nb in cands:
        if any(_valid(l, text[na:nb]) for l in CHECKSUM):
            return na, nb
    return a, b


# a whole dotted-quad or colon-hex token; the tagger splits long IPv6 into GSTIN/VEHICLE_REG pieces
IP_TOKEN = re.compile(r"(?<![\w:.])(?:(?:\d{1,3}\.){3}\d{1,3}|[0-9a-fA-F]{1,4}(?::[0-9a-fA-F]{0,4}){3,7})(?![\w:]|\.\w)")


def snap_ip(text, spans):
    """Replace tagger pieces that fall inside a valid IP token with the whole token."""
    out = list(spans)
    for m in IP_TOKEN.finditer(text):
        if not _ip(m.group()):
            continue
        hit = [s for s in out if s[0] not in FREE and s[1] < m.end() and s[2] > m.start()]
        if hit:
            out = [s for s in out if s not in hit] + [("IP_ADDRESS", m.start(), m.end())]
    return out


def postprocess(text, spans):
    out = []
    for lab, a, b in sorted(snap_ip(text, spans), key=lambda s: (s[1], -s[2])):
        if lab not in FREE:  # grow to whole word so a split never cuts a character group
            while a > 0 and _word_char(text[a - 1]): a -= 1
            while b < len(text) and _word_char(text[b]): b += 1
        whole = lambda s: len(s[3]) == 1 and _valid(s[0], text[s[1]:s[2]])
        if out and lab not in FREE and out[-1][0] not in FREE and SEP.fullmatch(text[out[-1][2]:a]) \
                and not (whole(out[-1]) and _valid(lab, text[a:b])):  # two complete IDs in a row stay apart
            prev = out.pop()
            pieces = prev[3] + [(lab, b - a)]
            out.append([prev[0], prev[1], max(b, prev[2]), pieces])
        elif out and a < out[-1][2]:  # overlap with a free-text span: keep the earlier one
            continue
        else:
            out.append([lab, a, b, [(lab, b - a)]])
    res = []
    for i, (lab, a, b, pieces) in enumerate(out):
        lo = out[i - 1][2] if i else 0
        hi = out[i + 1][1] if i + 1 < len(out) else len(text)
        if lab not in FREE:
            a, b = extend(text, a, b, lo, hi)
        if len(pieces) > 1:  # majority label by characters covered
            w = {}
            for l, n in pieces: w[l] = w.get(l, 0) + n
            lab = max(w, key=w.get)
        res.append((relabel(lab, text[a:b]), a, b))
    return res
