"""The deterministic rules: numbers, types, quotes, placeholders."""

from __future__ import annotations

import pytest

from sentinel import KnowledgePack
from sentinel.textrules import (
    TextRuleConfig,
    find_certainty_breaches,
    mask,
    scan_field,
)


# --------------------------------------------------------------------------- #
# Numbers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text", [
    "Porosity is 34.1% here.",
    "The value is 0.271.",
    "Measured at 35 degrees.",
    "A span of 1.4 was recorded.",
])
def test_literal_numbers_are_caught(text):
    _, hits = scan_field(text)
    assert any(h.kind == "digit" for h in hits), text


@pytest.mark.parametrize("text", [
    "The pore fraction is twice the other value.",
    "Roughly half of the surface is covered.",
    "It is three times larger.",
    "A couple of regions stand out.",
])
def test_number_words_are_caught(text):
    _, hits = scan_field(text)
    assert any(h.kind == "number_word" for h in hits), text


@pytest.mark.parametrize("text", [
    "Porosity is {stats.BSE.porosity.value}, close to the type {pred_type} profile.",
    "The difference is {stats.BSE.porosity.delta_vs_A} against type {runner_up}.",
    "All three detectors predict the same type.",
    "Cracks appear as dark ridges, measured with D50 as the reference size.",
    "This matches the published behaviour [S3].",
    "The chemistry is NMC811 in both products.",
    "Step 4 produced the prediction.",
    "Features come from DINOv3 with a ViT-B/16 backbone.",
])
def test_allowed_text_passes(text):
    _, hits = scan_field(text)
    errors = [h for h in hits if h.kind not in {"ordinal_word_warning"}]
    assert not errors, f"{text} -> {[(h.kind, h.text) for h in errors]}"


def test_placeholders_are_masked_not_removed():
    masked = mask("Porosity is {stats.BSE.porosity.value} exactly.")
    assert masked.placeholder_names() == ["stats.BSE.porosity.value"]
    assert "stats" not in masked.masked


def test_digits_inside_placeholders_do_not_trip_the_scan():
    _, hits = scan_field("The size is {stats.BSE.d50.value} across.")
    assert not [h for h in hits if h.kind == "digit"]


def test_unbalanced_brace_is_caught():
    _, hits = scan_field("Porosity is {stats.BSE.porosity.value exactly.")
    assert any(h.kind == "brace" for h in hits)


def test_extra_allow_patterns_are_honoured():
    config = TextRuleConfig().with_extra(patterns=(r"\bISO\s*\d{4,5}\b",))
    _, hits = scan_field("Measured per ISO 20776 conventions.", config)
    assert not [h for h in hits if h.kind == "digit"]


# --------------------------------------------------------------------------- #
# Type immutability
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text", [
    "This sample is type A.",
    "It resembles type C more closely.",
    "A type-B electrode looks like this.",
])
def test_bare_type_letters_are_caught(text):
    _, hits = scan_field(text)
    assert any(h.kind == "bare_type" for h in hits), text


def test_fixed_phrase_naming_all_types_is_allowed():
    _, hits = scan_field("The classifier distinguishes types A, B and C.")
    assert not [h for h in hits if h.kind == "bare_type"]


def test_type_placeholders_are_allowed():
    _, hits = scan_field("It is type {pred_type}, not type {runner_up}.")
    assert not [h for h in hits if h.kind == "bare_type"]


# --------------------------------------------------------------------------- #
# Attention and certainty
# --------------------------------------------------------------------------- #
def test_attention_is_not_evidence():
    _, hits = scan_field("The attention map focuses on the particle edges.")
    assert any(h.kind == "attention" for h in hits)


def test_attention_allowed_in_a_caveat():
    _, hits = scan_field(
        "The attention map is not class-specific and is shown for context only.",
        allow_attention=True,
    )
    assert not [h for h in hits if h.kind == "attention"]


@pytest.mark.parametrize("label", ["high", "medium", "low"])
def test_proof_language_banned_at_every_label(label):
    hits = find_certainty_breaches("The microstructure proves the type.", label)
    assert hits, label


def test_clearly_allowed_at_high_but_not_medium():
    assert not find_certainty_breaches("The evidence clearly shows the pattern.", "high")
    assert find_certainty_breaches("The evidence clearly shows the pattern.", "medium")


def test_strongly_banned_only_at_low():
    assert not find_certainty_breaches("This strongly indicates the type.", "medium")
    assert find_certainty_breaches("This strongly indicates the type.", "low")


# --------------------------------------------------------------------------- #
# Quote matching
# --------------------------------------------------------------------------- #
PACK_TEXT = """\
## [S1] Detector physics
- url: https://example.invalid/s1

The backscattered electron yield rises with the mean atomic number, so dense
oxide particles appear brighter than carbon-rich binder.

## [S2] Processing
- url: https://example.invalid/s2

Calendering reduces total porosity.
"""


@pytest.fixture
def small_pack():
    return KnowledgePack.from_text(PACK_TEXT)


def test_quote_matches_across_a_line_wrap(small_pack):
    check = small_pack.verify_quote(
        "S1",
        "The backscattered electron yield rises with the mean atomic number, so "
        "dense oxide particles appear brighter than carbon-rich binder.",
    )
    assert check.ok, check.reason


def test_quote_matches_through_curly_punctuation(small_pack):
    check = small_pack.verify_quote("S2", "Calendering reduces total porosity.")
    assert check.ok, check.reason


def test_paraphrase_is_rejected_with_a_suggestion(small_pack):
    check = small_pack.verify_quote("S2", "Calendering lowers the porosity.")
    assert not check.ok
    assert check.suggestion and "Calendering reduces total porosity" in check.suggestion


def test_unknown_source_is_rejected(small_pack):
    check = small_pack.verify_quote("S9", "anything")
    assert not check.ok
    assert "not in the knowledge pack" in (check.reason or "")


def test_empty_quote_is_rejected(small_pack):
    assert not small_pack.verify_quote("S1", "   ").ok


def test_undefined_markers_are_reported(small_pack):
    assert small_pack.undefined_markers("As shown [S1] and [S7] and [S9].") == ["S7", "S9"]


def test_duplicate_source_ids_are_rejected():
    with pytest.raises(ValueError, match="duplicate source"):
        KnowledgePack.from_text("## [S1] a\n\nx\n\n## [S1] b\n\ny\n")


def test_search_returns_quotable_snippets(small_pack):
    hits = small_pack.search("calendering porosity")
    assert hits and hits[0]["source"] == "S2"
    assert small_pack.verify_quote("S2", hits[0]["snippet"]).ok


def test_pack_version_changes_with_content(small_pack):
    other = KnowledgePack.from_text(PACK_TEXT + "\n## [S3] More\n\nText.\n")
    assert small_pack.version != other.version


# --------------------------------------------------------------------------- #
# Regressions found by the demo run
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text", [
    "The label is medium rather than a lower one.",
    "One of the images carries a flag.",
    "This is the one that separates the types.",
    "One or more detectors may be missing.",
])
def test_one_as_a_pronoun_is_not_a_quantity(text):
    """Blocking bare "one" rejected valid drafts over nothing."""
    _, hits = scan_field(text)
    assert not [h for h in hits if h.kind == "number_word"], text


@pytest.mark.parametrize("text", [
    "The particles are about one micron across.",
    "It sits one standard deviation from the profile.",
    "Porosity is one percentage point higher.",
])
def test_one_before_a_unit_is_still_a_quantity(text):
    _, hits = scan_field(text)
    assert [h for h in hits if h.kind == "number_word"], text


def test_one_and_a_half_times_is_still_caught():
    _, hits = scan_field("The span is roughly one and a half times the mean.")
    assert [h for h in hits if h.kind == "number_word"]


@pytest.mark.parametrize("text", [
    "The separation between the two types is clear.",
    "It has the highest probability of the three.",
    "All three detectors agree.",
    "Both types overlap here.",
    "Across the three profiles the pattern holds.",
])
def test_a_cardinal_after_the_is_a_reference_not_a_count(text):
    _, hits = scan_field(text)
    assert not [h for h in hits if h.kind == "number_word"], text


@pytest.mark.parametrize("text", [
    "Two detectors predict this type.",
    "Three statistics separate them.",
    "Two of the samples were excluded.",
])
def test_a_bare_cardinal_is_still_a_count(text):
    _, hits = scan_field(text)
    assert [h for h in hits if h.kind == "number_word"], text


@pytest.mark.parametrize("text", [
    "This matches the literature [S3].",
    "This matches the literature (S3).",
    "Cracking can come from either route (S4).",
    "Both sources agree [S1, S2].",
    "As S5 notes, the measure is comparative.",
])
def test_citation_markers_are_not_numbers_in_any_bracket_style(text):
    _, hits = scan_field(text)
    assert not [h for h in hits if h.kind == "digit"], text


@pytest.mark.parametrize("marker", ["[S99]", "(S99)", "S99"])
def test_an_invented_source_is_still_caught_in_any_style(small_pack, marker):
    """Excusing the digit must not excuse the citation."""
    assert small_pack.undefined_markers(f"As shown {marker}.") == ["S99"]


@pytest.mark.parametrize("text", [
    "The D50 is close to the profile.",
    "The d50 is close to the profile.",         # the contract spells it lowercase
    "Spread runs from d10 to d90.",
])
def test_particle_percentiles_are_allowed_in_either_case(text):
    _, hits = scan_field(text)
    assert not [h for h in hits if h.kind == "digit"], text
