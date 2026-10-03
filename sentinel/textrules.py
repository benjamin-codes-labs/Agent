"""Deterministic text rules behind the validator (proposal section 6.4).

The proposal's rule is "no numbers outside placeholders", with an allow-list for
citation markers, chemical formulas, D10/D50/D90, units, step numbers and fixed
phrases, and with number *words* such as "three times" or "half" also caught.

Implementation is a masking pass: placeholders and allow-listed patterns are
blanked out of the string, and whatever digits or number words survive are
violations. Masking (rather than matching) is what makes the allow-list
composable -- the fixed phrase "types A, B and C" is removed before both the
number scan and the bare-type-letter scan, so one entry serves both rules.

Everything here is configurable because the right allow-list is discovered by
running real explainer output through it, not by guessing in advance. Tune
:class:`TextRuleConfig` on day one against actual output; do not loosen a rule
to make a failing explanation pass.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace

PLACEHOLDER = re.compile(r"\{([^{}]*)\}")
MASK_CHAR = "░"

# --------------------------------------------------------------------------- #
# Default allow-lists
# --------------------------------------------------------------------------- #
#: Patterns whose digits are structural, not measurements.
DEFAULT_ALLOW_PATTERNS: tuple[str, ...] = (
    # Region names are identifiers, not quantities. Without this the validator
    # rejects its own template output, since every region is named "<det>
    # region <n>". Prose region names are checked for existence separately,
    # by REGION003, so masking them here does not let a made-up region through.
    r"\b(?:BSE|ETD|InLens)\s+region\s+\d+\b",
    # Citation markers, in any bracket style the model reaches for. The digit
    # is a source ID, never a measurement. Existence is still enforced:
    # knowledge.CITATION_MARKER matches the same shapes, so an invented (S99)
    # is caught by CITE003.
    r"[\[(]S\d+(?:\s*[,;]\s*S\d+)*[\])]",
    r"\bS\d+\b",
    # Particle-size percentiles. Case-insensitive on purpose: the contract
    # names the statistic "d50", so the model writes it lowercase as often as
    # not, and a case-sensitive rule turned the "50" into a loose number.
    r"\b[Dd](?:10|50|90)\b",
    r"\bStep\s+[1-7]\b",                            # pipeline step numbers
    r"\bDINOv[23]\b",                               # backbone names
    r"\bViT-[BSL](?:/\d+)?\b",
    r"\bXGBoost\b",
    r"\b(?:NMC|NCA|LFP|LCO|LMO|LTO|LMFP|LNMO)[-\s]?\d{0,4}\b",   # cathode chemistries
    r"\b(?:Li|Na)[A-Za-z]{1,6}\d[A-Za-z0-9]*\b",   # LiFePO4, Li4Ti5O12
    r"\bC\d{1,2}\b",                                # graphite C6
    r"\b(?:µm|um|nm|mm)\s*[23]\b",                  # area/volume units written flat
    r"\bkV\b",
    # "one" as a determiner or pronoun is not a quantity claim ("one of the
    # images", "one or more"). Leaving it blocked fails almost every natural
    # sentence and burns the retry budget on nothing; the quantitative uses that
    # actually matter are the multipliers and fractions below, which stay
    # blocked. Other cardinals are NOT excused: "two of the three detectors"
    # must use {cls.detector_agreement}.
    r"\bone of\b",
    r"\bone or more\b",
    r"\bno one\b",
    r"\bthe one\b",
    # A cardinal after a definite article refers to a set already named -- "the
    # two types" means {pred_type} and {runner_up}, "of the three" means the
    # three battery types. That is a reference, not a measurement. A bare
    # cardinal in subject position still is one: "two detectors predict type B"
    # stays blocked, and must use {cls.detector_agreement}.
    r"\b(?:the|both|between the|of the|across the|among the|all)\s+"
    r"(?:two|three|four|five)\b",
)

#: Phrases that may appear verbatim even though they contain a number or a bare
#: type letter. These describe the *system*, not the measurement.
DEFAULT_ALLOW_PHRASES: tuple[str, ...] = (
    "all three detectors",
    "the three detectors",
    "three detectors",
    "all three types",
    "the three types",
    "three types",
    "types A, B and C",
    "types A, B, and C",
    "type A, B or C",
    "one of the three",
    "one another",
    "on the one hand",
    "on the other hand",
    "no single",
    "first principles",
)

#: Number words that count as numbers. Cardinals, fractions and multipliers.
#:
#: "one" is deliberately absent. It is a pronoun and a determiner far more often
#: than a quantity ("one of the images", "rather than a lower one"), and leaving
#: it in rejected valid drafts over nothing. Its genuinely quantitative uses are
#: covered two ways: a fraction or multiplier alongside it is caught on its own
#: ("one and a half times" trips on "half"), and "one <unit>" is caught by
#: DEFAULT_QUANTITY_PHRASES below. Every other cardinal stays blocked, so
#: "two of the three detectors" must use {cls.detector_agreement}.
DEFAULT_NUMBER_WORDS: tuple[str, ...] = (
    "zero", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty",
    "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred",
    "thousand", "million",
    "half", "halves", "third", "thirds", "quarter", "quarters", "fifth",
    "fifths", "two-thirds", "three-quarters",
    "twice", "thrice", "double", "doubled", "triple", "tripled", "quadruple",
    "twofold", "two-fold", "threefold", "three-fold", "fourfold", "tenfold",
    "dozen", "couple",
)

#: Quantity phrases that survive the word list above because their number word
#: is excused. "one micron" is a measurement; "one of the images" is not.
DEFAULT_QUANTITY_PHRASES: tuple[str, ...] = (
    r"\bone\s+(?:micron|microns|µm|um|nm|mm|percent|per\s+cent|percentage\s+point"
    r"|percentage\s+points|pp|fold|order\s+of\s+magnitude|standard\s+deviation"
    r"|standard\s+deviations|sd)\b",
    r"\ba\s+(?:half|third|quarter)\s+of\b",
)

#: Ordinals are numbers too, but blocking them makes ordinary prose impossible
#: ("the first region"). Warned about rather than failed, by default.
DEFAULT_ORDINAL_WORDS: tuple[str, ...] = (
    "first", "second", "fourth", "fifth", "sixth", "seventh", "eighth",
    "ninth", "tenth",
)

#: Wording that overstates what a probabilistic classifier can deliver. Banned
#: at every confidence label -- Sentinel never proves a type.
CERTAINTY_ALWAYS_BANNED: tuple[str, ...] = (
    "proves", "proven", "proof that", "certainly", "definitely", "undoubtedly",
    "without doubt", "beyond doubt", "conclusively", "unambiguously",
    "guarantees", "guaranteed", "100% certain", "no doubt",
    "must be type", "can only be", "confirms that this",
)

#: Additionally banned when the confidence label is not "high".
CERTAINTY_BANNED_BELOW_HIGH: tuple[str, ...] = (
    "clearly shows", "clearly indicates", "unmistakab", "decisive",
    "definitive", "establishes that", "leaves no",
)

#: Additionally banned when the label is "low".
CERTAINTY_BANNED_AT_LOW: tuple[str, ...] = (
    "strongly", "confident", "confidently", "reliably", "robustly",
)

#: At least one of these must appear somewhere when the label is "low".
DEFAULT_LOW_CONFIDENCE_HEDGES: tuple[str, ...] = (
    "tentative", "provisional", "cannot be distinguished", "weak", "uncertain",
    "not reliable", "should be reviewed", "low confidence", "inconclusive",
    "further imaging", "treat with caution", "cannot reliably",
)

#: "Attention is not evidence" (proposal 5.1). Banned in evidential prose.
ATTENTION_WORDS: tuple[str, ...] = ("attention map", "attention-map", "attention")

BARE_TYPE_LETTER = re.compile(r"\btypes?[-\s]+([ABC])\b")
REGION_MENTION = re.compile(r"\b(BSE|ETD|InLens)\s+region\s+(\d+)\b")


def find_region_mentions(text: str) -> list[tuple[str, int, int]]:
    """Region names named in prose, as ``(canonical name, start, end)``.

    Region names are masked out of the number scan as identifiers, so they are
    checked for existence here instead -- otherwise "InLens region 4" would
    sail through on a battery that has only one InLens region.
    """
    return [
        (f"{m.group(1)} region {int(m.group(2))}", m.start(), m.end())
        for m in REGION_MENTION.finditer(text)
    ]


@dataclass(frozen=True)
class TextRuleConfig:
    allow_patterns: tuple[str, ...] = DEFAULT_ALLOW_PATTERNS
    allow_phrases: tuple[str, ...] = DEFAULT_ALLOW_PHRASES
    number_words: tuple[str, ...] = DEFAULT_NUMBER_WORDS
    quantity_phrases: tuple[str, ...] = DEFAULT_QUANTITY_PHRASES
    ordinal_words: tuple[str, ...] = DEFAULT_ORDINAL_WORDS
    ordinals_are_errors: bool = False
    low_confidence_hedges: tuple[str, ...] = DEFAULT_LOW_CONFIDENCE_HEDGES
    extra_allow_patterns: tuple[str, ...] = ()
    extra_allow_phrases: tuple[str, ...] = ()

    def with_extra(
        self, patterns: tuple[str, ...] = (), phrases: tuple[str, ...] = ()
    ) -> "TextRuleConfig":
        return replace(
            self,
            extra_allow_patterns=self.extra_allow_patterns + patterns,
            extra_allow_phrases=self.extra_allow_phrases + phrases,
        )

    def all_patterns(self) -> tuple[str, ...]:
        return self.allow_patterns + self.extra_allow_patterns

    def all_phrases(self) -> tuple[str, ...]:
        return self.allow_phrases + self.extra_allow_phrases


@dataclass
class Hit:
    """One offending span."""

    kind: str
    text: str
    start: int
    end: int
    detail: str = ""


@dataclass
class MaskedText:
    """A string with placeholders and allow-listed spans blanked out."""

    original: str
    masked: str
    placeholders: list[tuple[str, int, int]] = field(default_factory=list)

    def placeholder_names(self) -> list[str]:
        return [name for name, _, _ in self.placeholders]

    def context(self, start: int, end: int, width: int = 34) -> str:
        lo = max(0, start - width)
        hi = min(len(self.original), end + width)
        prefix = "..." if lo > 0 else ""
        suffix = "..." if hi < len(self.original) else ""
        return f"{prefix}{self.original[lo:hi]}{suffix}"


def _blank(text: str, start: int, end: int) -> str:
    return text[:start] + MASK_CHAR * (end - start) + text[end:]


def mask(text: str, config: TextRuleConfig | None = None) -> MaskedText:
    """Blank out placeholders and every allow-listed span."""
    config = config or TextRuleConfig()
    masked = text
    placeholders: list[tuple[str, int, int]] = []

    for match in PLACEHOLDER.finditer(text):
        placeholders.append((match.group(1).strip(), match.start(), match.end()))
        masked = _blank(masked, match.start(), match.end())

    # Phrases first: they are the coarsest and may contain pattern matches.
    for phrase in sorted(config.all_phrases(), key=len, reverse=True):
        for match in re.finditer(re.escape(phrase), masked, re.IGNORECASE):
            masked = _blank(masked, match.start(), match.end())

    for pattern in config.all_patterns():
        for match in re.finditer(pattern, masked):
            masked = _blank(masked, match.start(), match.end())

    return MaskedText(original=text, masked=masked, placeholders=placeholders)


# --------------------------------------------------------------------------- #
# Individual scans
# --------------------------------------------------------------------------- #
def find_unbalanced_braces(text: str) -> list[Hit]:
    """A stray brace means a placeholder will leak to the screen unrendered."""
    stripped = PLACEHOLDER.sub(lambda m: MASK_CHAR * len(m.group(0)), text)
    return [
        Hit("brace", ch, i, i + 1, "unbalanced brace; a placeholder is malformed")
        for i, ch in enumerate(stripped)
        if ch in "{}"
    ]


def find_digits(masked: MaskedText) -> list[Hit]:
    """Any surviving digit run is a number the code did not supply."""
    hits: list[Hit] = []
    for match in re.finditer(r"\d[\d.,:/]*", masked.masked):
        if MASK_CHAR in match.group(0):
            continue
        hits.append(Hit(
            "digit",
            masked.original[match.start() : match.end()],
            match.start(),
            match.end(),
            "a literal number outside a placeholder; cite a fact ID instead",
        ))
    return hits


def find_number_words(masked: MaskedText, config: TextRuleConfig) -> list[Hit]:
    hits: list[Hit] = []
    for pattern in config.quantity_phrases:
        for match in re.finditer(pattern, masked.masked, re.IGNORECASE):
            hits.append(Hit(
                "number_word",
                masked.original[match.start() : match.end()],
                match.start(),
                match.end(),
                f"{match.group(0)!r} states a measured quantity in words; "
                "use a fact placeholder",
            ))
    words = sorted(config.number_words, key=len, reverse=True)
    for word in words:
        for match in re.finditer(rf"\b{re.escape(word)}\b", masked.masked, re.IGNORECASE):
            hits.append(Hit(
                "number_word",
                masked.original[match.start() : match.end()],
                match.start(),
                match.end(),
                f"the number word {word!r} states a quantity; use a fact placeholder",
            ))
    kind = "ordinal_word" if config.ordinals_are_errors else "ordinal_word_warning"
    for word in config.ordinal_words:
        for match in re.finditer(rf"\b{re.escape(word)}\b", masked.masked, re.IGNORECASE):
            hits.append(Hit(
                kind,
                masked.original[match.start() : match.end()],
                match.start(),
                match.end(),
                f"the ordinal {word!r} is a number word",
            ))
    return hits


def find_bare_type_letters(masked: MaskedText) -> list[Hit]:
    """Keep the predicted type immutable: types are named by placeholder only."""
    return [
        Hit(
            "bare_type",
            masked.original[match.start() : match.end()],
            match.start(),
            match.end(),
            "name a type with {pred_type} or {runner_up}, never a literal letter, "
            "so the explanation cannot contradict Step 4",
        )
        for match in BARE_TYPE_LETTER.finditer(masked.masked)
    ]


def find_attention_as_evidence(text: str) -> list[Hit]:
    """The attention map is not class-specific, so it is never evidence."""
    hits: list[Hit] = []
    low = text.lower()
    for word in ATTENTION_WORDS:
        start = 0
        while (idx := low.find(word, start)) != -1:
            hits.append(Hit(
                "attention",
                text[idx : idx + len(word)],
                idx,
                idx + len(word),
                "the attention map is not class-specific and must not be offered "
                "as evidence; cite an evidence region instead",
            ))
            start = idx + len(word)
            break  # one hit per word is enough to fail the field
    return hits


def find_certainty_breaches(text: str, confidence: str) -> list[Hit]:
    """Certainty wording must match the confidence label (proposal 6.3)."""
    banned = list(CERTAINTY_ALWAYS_BANNED)
    if confidence != "high":
        banned += list(CERTAINTY_BANNED_BELOW_HIGH)
    if confidence == "low":
        banned += list(CERTAINTY_BANNED_AT_LOW)

    hits: list[Hit] = []
    low = text.lower()
    for phrase in banned:
        idx = low.find(phrase)
        if idx != -1:
            hits.append(Hit(
                "certainty",
                text[idx : idx + len(phrase)],
                idx,
                idx + len(phrase),
                f"{phrase!r} overstates a {confidence}-confidence prediction",
            ))
    return hits


def missing_low_confidence_hedge(text: str, config: TextRuleConfig) -> bool:
    low = text.lower()
    return not any(h in low for h in config.low_confidence_hedges)


def scan_field(
    text: str,
    config: TextRuleConfig | None = None,
    *,
    allow_attention: bool = False,
    confidence: str | None = None,
) -> tuple[MaskedText, list[Hit]]:
    """Run every lexical scan over one free-text field."""
    config = config or TextRuleConfig()
    masked = mask(text, config)
    hits: list[Hit] = []
    hits.extend(find_unbalanced_braces(text))
    hits.extend(find_digits(masked))
    hits.extend(find_number_words(masked, config))
    hits.extend(find_bare_type_letters(masked))
    if not allow_attention:
        hits.extend(find_attention_as_evidence(text))
    if confidence:
        hits.extend(find_certainty_breaches(text, confidence))
    hits.sort(key=lambda h: h.start)
    return masked, hits
