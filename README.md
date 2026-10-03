# Sentinel v3 — Step 6, the explainer agent

Implements proposal section 6 (Benjamin's piece): the agentic workflow that turns
a finished classification into a checked, cited explanation.

```
Step 3/4/5 output (JSON contract)
   │
   ├─ evidence builder ............ code      facts.py      every number gets an ID
   ├─ explainer ................... Sonnet 5.5 + vision   explainer.py
   ├─ validator ................... code      validator.py  numbers, IDs, quotes, type
   ├─ grounding critic ............ Haiku 4.5 critic.py     attention, certainty, support
   ├─ retry ×2, then template ..... code      workflow.py / template.py
   └─ renderer .................... code      renderer.py   placeholders → real numbers
                                                            → explanations/{battery}.json
```

```bash
pip install pydantic jinja2 anthropic pillow     # pillow optional
python -m pytest -q                              # 99 tests, no API key needed

python -m sentinel.cli selftest  examples/BAT-07.json --pack knowledge/pack.md
python -m sentinel.cli explain   examples/BAT-07.json --pack knowledge/pack.md --template-only
python -m sentinel.cli facts     examples/BAT-07.json
ANTHROPIC_API_KEY=... python -m sentinel.cli explain examples/BAT-07.json \
    --pack knowledge/pack.md --images --image-root runs --out out.json

modal deploy sentinel/modal_app.py
modal run sentinel/modal_app.py::explain_one --battery-id BAT-07 --run-id R1
```

## What is faithful to the proposal

| Proposal | Where |
|---|---|
| Fixed workflow, not a free agent | `workflow.py` |
| Fact sheet with an ID per number (`stats.BSE.porosity.z_B`) | `facts.py` |
| Sonnet 5.5, one call, vision, images before text, pack cached | `explainer.py`, `prompts.py` |
| No forced `tool_choice`, no `temperature` for Sonnet 5.5 | `ExplainerConfig`, asserted in tests |
| Validator: placeholders, region IDs, no loose numbers, exact quotes, type unchanged | `validator.py`, `textrules.py` |
| Haiku 4.5 grounding critic | `critic.py` |
| Retry ×2, then Jinja2 template | `workflow.py`, `template.py` |
| Renderer fills placeholders | `renderer.py` |
| One pydantic model, `classification` read-only to Step 6 | `contract.py` |
| Modal `explain_battery(battery_id)` writing `explanations/{battery}.json` | `modal_app.py` |
| v2 read-only tools with a step limit | `tools.py` |
| Cut order: critic first | `WorkflowConfig(use_critic=False)` / `NullCritic` |

## Six places this deviates, and why

**1. Derived comparison facts.** The proposal forbids numbers outside
placeholders, but "8 points above type A's mean" is the sentence the explainer
naturally wants and no fact ID existed for it. It would invent the number, fail
the validator, burn both retries and fall back to the template *every time*.
The builder now precomputes `…delta_vs_A` per statistic and `…pore_delta` per
region. This was the highest-risk gap in the design.

**2. A content floor (`CONTENT*`, `CITE002`).** Every other rule is satisfied
most easily by writing nothing — no numbers, no citations, no regions passes
them all. With a retry loop pushing towards "whatever the validator accepts",
the cheapest escape is a vacuous paragraph. The floor requires ≥2 evidence
items, ≥1 grounded in a measured fact, ≥1 citation, and the runner-up addressed.
The selftest's "evidence deleted (gaming)" row demonstrates it. Relevant to the
"knows when it's reward hacking" framing more than anything else here.

**3. `display` strings.** Each fact carries the one string that will ever be
shown, computed once. Without this the text rounds one way and the JSON another,
and the audit trail disagrees with the screen.

**4. Critic opinion vs critic authority.** `CriticVerdict.passed` (did it block)
is separate from `declared_passed` (did it object). With `blocking=False` the
explanation ships but `explanation.critic_passed` is still `False`, so a
non-blocking critic's objection is visible rather than laundered into a pass.
Entailment judgement on scientific text is weak, so the critic is a one-way
gate: it can fail an explanation, never pass one the code validator failed.

**5. Loop guards.** Monotone progress on `(errors, warnings)`, plus oscillation
detection by hashing the sorted `(code, loc, observed)` multiset. Identical
failures twice ends the loop early instead of paying for a third identical
call. The conversation is replayed so each retry is a diff, not a fresh draft.

**6. Seeded-defect selftest.** "100% of delivered explanations pass the
validator" is satisfied trivially by a validator that checks nothing. The
selftest answers the other direction — when an explanation *is* wrong, does the
checker notice — and prints 16/16 in about a second, with no API call.

## Two bugs the tests caught, worth knowing about

- **Region names contain digits.** `"BSE region 1"` tripped the no-literal-numbers
  rule, so the validator rejected its own template output. Region names are now
  masked as identifiers, and prose region names are checked for existence
  separately (`REGION003`) so a made-up `"InLens region 4"` is still caught.
- **The retry conversation began with an assistant turn**, which is an API
  error. The full message list is now carried forward, which also keeps the
  images in context on retries without re-uploading them.

## Tuning on the day

`TextRuleConfig` is the knob. The number-word lexicon is strict by design and
*will* produce false positives on real output; `"one of"` is already excused
because blocking it fails almost every natural sentence, while other cardinals
stay blocked (`"two of the three detectors"` must use
`{cls.detector_agreement}`). Add allow-patterns rather than deleting rules:

```python
WorkflowConfig(validator=ValidatorConfig(
    text_rules=TextRuleConfig().with_extra(patterns=(r"\bISO\s*\d{4,5}\b",)),
))
```

Watch for: a statistic whose label contains a digit; chemistry names beyond the
cathode list; and ordinals, which are warnings not errors (`ordinals_are_errors=True`
to tighten).

### What live runs actually tripped on

Every one of these was a false positive that cost a retry, and each is now
excused. The pattern is the same throughout — **a number word that refers to
something rather than measuring it**:

| Model wrote | Why it is not a measurement |
|---|---|
| "the medium label rather than a lower **one**" | pronoun |
| "**one of** the images carries a flag" | determiner |
| "the separation between **the two** types" | definite reference to {pred_type} and {runner_up} |
| "the highest probability **of the three**" | definite reference to the three types |
| "about **thirty** batteries each" | the §3.2 caveat, which now has `{profiles.batteries_per_type}` |

Counts still fail, correctly: "**two** detectors predict this type" must use
`{cls.detector_agreement}`. If you add an allow-rule, add a test beside
`test_a_bare_cardinal_is_still_a_count` proving the quantitative form still
fails — that is what stops the allow-list eating the rule.

Also learned the hard way: **the explainer tool is off by default**
(`offer_tool=False`). `tool_choice` cannot be forced on this model, so the tool
is only a suggestion, and when the model took it its arguments came back twice
as pseudo-XML with `evidence` as a string. The JSON-in-text path has parsed
first time on every live attempt.

## Two things to say honestly on stage

The validator checks that a quote **exists and is unaltered**, not that it is
**apt** — "the quote is real but about something else" is case C for the critic,
and that is model judgement, not proof. And `knowledge/pack.md` is a plumbing
stub with `UNVERIFIED STUB` on every source: it exists so the quote matcher has
something to run against. Nothing in it should be shown as cited literature
until Zoe's real pack replaces it.

## Open against the proposal's `[TBD]`s

- **Missing detector.** Handled (`available_detectors`, confidence drops), but
  the fusion weights themselves live in Step 4.
- **Statistic labels** are read from `StatisticRecord.label`; Gabriel's Step 3
  should populate it or the prose falls back to the raw name.
- **Image co-registration** (organiser question 3) does not affect Step 6: maps
  and local statistics are already per detector.
- `collect_images` reads the `maps.*_png` paths from the contract and skips
  files that are absent, so it is safe to run before Step 5 finishes.
