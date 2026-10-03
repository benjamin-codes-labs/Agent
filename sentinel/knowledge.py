"""Knowledge pack (proposal section 6.5) and deterministic quote checking.

The pack is markdown. Every source is a heading of the form ``## [S3] Title``
followed by optional ``- key: value`` metadata and then prose. The validator
requires that each citation quote appears *word for word* in its cited source,
so the matching here is the load-bearing part.

Matching policy
---------------
Exact-substring matching on raw markdown fails constantly for reasons that have
nothing to do with honesty: the pack is line-wrapped, Paperclip and PDF sources
emit curly quotes, en dashes and non-breaking spaces, and the model retypes
them as ASCII. So both sides are normalised first -- whitespace collapsed,
unicode punctuation folded to ASCII, soft hyphens dropped -- and then compared
case-sensitively. Nothing about the wording is allowed to change; only the
typography is forgiven.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .facts import Fact

SOURCE_HEADING = re.compile(r"^#{1,6}\s*\[(S\d+)\]\s*(.*?)\s*$", re.MULTILINE)
META_LINE = re.compile(r"^[-*]\s*([A-Za-z_][A-Za-z0-9_ ]*):\s*(.+?)\s*$")
#: Citation markers in prose, in any bracket style -- "[S3]", "(S3)", "S3".
#: Must stay in step with the allow-pattern in textrules, so that every shape
#: excused from the number rule is still checked for existence.
CITATION_MARKER = re.compile(r"(?:[\[(]\s*)?\bS(\d+)\b(?:\s*[\])])?")

_PUNCT_FOLD = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "′": "'", "″": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-",
    "—": "-", "―": "-", "−": "-",
    " ": " ", " ": " ", " ": " ", " ": " ", " ": " ",
    "­": "", "​": "", "﻿": "",
    "…": "...",
}


def normalise_for_match(text: str) -> str:
    """Fold typography and collapse whitespace; keep every word and its case."""
    text = unicodedata.normalize("NFKC", text)
    text = "".join(_PUNCT_FOLD.get(ch, ch) for ch in text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


@dataclass
class Source:
    """One cited source in the pack."""

    id: str
    title: str
    body: str
    metadata: dict[str, str] = field(default_factory=dict)
    fields: dict[str, Any] | None = field(default=None, repr=False)

    @property
    def url(self) -> str | None:
        for key in ("url", "link", "doi"):
            if key in self.metadata:
                return self.metadata[key]
        return None

    @property
    def normalised(self) -> str:
        return normalise_for_match(f"{self.title}\n{self.body}")

    def contains(self, quote: str) -> bool:
        return normalise_for_match(quote) in self.normalised

    def closest_sentence(self, quote: str) -> str | None:
        """Nearest real sentence, so a failed quote gets actionable feedback."""
        target = normalise_for_match(quote)
        if not target:
            return None
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", self.normalised) if s.strip()]
        if not sentences:
            return None
        best = max(sentences, key=lambda s: difflib.SequenceMatcher(None, target, s).ratio())
        ratio = difflib.SequenceMatcher(None, target, best).ratio()
        return best if ratio >= 0.55 else None


@dataclass
class QuoteCheck:
    ok: bool
    source_id: str
    quote: str
    reason: str | None = None
    suggestion: str | None = None


class KnowledgePack:
    """Parsed pack with source lookup, quote verification and plain search."""

    def __init__(self, sources: dict[str, Source], *, raw: str = "", path: Path | None = None,
                 context_facts: dict[str, Fact] | None = None):
        self.sources = sources
        self.raw = raw
        self.path = path
        self.context_facts = dict(context_facts or {})

    # -- construction ----------------------------------------------------- #
    @classmethod
    def from_text(cls, text: str, *, path: Path | None = None) -> "KnowledgePack":
        matches = list(SOURCE_HEADING.finditer(text))
        sources: dict[str, Source] = {}
        for i, match in enumerate(matches):
            sid, title = match.group(1), match.group(2)
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            block = text[start:end]
            metadata: dict[str, str] = {}
            body_lines: list[str] = []
            in_meta = True
            for line in block.splitlines():
                meta = META_LINE.match(line) if in_meta else None
                if meta:
                    metadata[meta.group(1).strip().lower().replace(" ", "_")] = meta.group(2)
                    continue
                if in_meta and line.strip():
                    in_meta = False
                body_lines.append(line)
            if sid in sources:
                raise ValueError(f"duplicate source id {sid} in the knowledge pack")
            sources[sid] = Source(
                id=sid, title=title, body="\n".join(body_lines).strip(), metadata=metadata
            )
        return cls(sources, raw=text, path=path)

    @classmethod
    def from_path(cls, path: str | Path) -> "KnowledgePack":
        p = Path(path)
        if p.is_dir():
            parts = [f.read_text(encoding="utf-8") for f in sorted(p.glob("**/*.md"))]
            return cls.from_text("\n\n".join(parts), path=p)
        return cls.from_text(p.read_text(encoding="utf-8"), path=p)

    @classmethod
    def from_project(cls, root: str | Path) -> "KnowledgePack":
        root = Path(root)
        reference_path = root / "SiC_SEM_reference_verified.json"
        batch_path = root / "all_batch_comparisons.json"
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        batches = json.loads(batch_path.read_text(encoding="utf-8"))
        if not isinstance(reference, dict) or not isinstance(reference.get("features"), list):
            raise ValueError(f"{reference_path.name} must contain a features list")
        if not isinstance(batches, dict) or not isinstance(batches.get("detectors"), dict):
            raise ValueError(f"{batch_path.name} must contain a detectors object")
        raw = json.dumps({reference_path.name: reference, batch_path.name: batches}, ensure_ascii=False, allow_nan=False)
        pack = cls({}, raw=raw, path=root)
        pack._add_record("SEM reference scope and verification limits", reference.get("meta", {}),
                         "scope", f"{reference_path.name}#/meta")
        references = {work["id"]: work for work in reference.get("references", [])}
        for section in ("features", "properties", "architectures", "imaging_modes", "artifacts"):
            for index, record in enumerate(reference.get(section, [])):
                if not isinstance(record, dict):
                    raise ValueError(f"{reference_path.name}: {section}[{index}] must be an object")
                works = record.get("verification", {}).get("supporting_works", [])
                enriched = {**record, "supporting_references": [references[wid] for wid in works if wid in references]}
                pack._add_record(f"{section}: {record.get('name', record.get('id', index))}", enriched,
                                 "reference", f"{reference_path.name}#/{section}/{index}")
        pack._add_record("Batch comparison scope, exclusions and limitations",
                         {key: value for key, value in batches.items() if key != "detectors"},
                         "scope", f"{batch_path.name}#/")
        for detector, data in batches["detectors"].items():
            pack._add_record(f"{detector.upper()} batch comparison limitations",
                             {key: value for key, value in data.items() if key not in {"batches", "comparisons"}},
                             "batch", f"{batch_path.name}#/detectors/{detector}")
            for batch, summary in data.get("batches", {}).items():
                compact = {key: value for key, value in summary.items() if key not in {"images", "input_files", "phases"}}
                compact["batch"] = batch
                compact["phases"] = {
                    name: {key: value for key, value in phase.items() if key != "images"}
                    for name, phase in summary.get("phases", {}).items()
                }
                pack._add_record(f"{detector.upper()} aggregate measurements for {batch}", compact,
                                 "batch", f"{batch_path.name}#/detectors/{detector}/batches/{batch}")
            for index, comparison in enumerate(data.get("comparisons", [])):
                pack._add_record(f"{detector.upper()} {comparison['lot_b']} minus {comparison['lot_a']}",
                                 comparison, "comparison", f"{batch_path.name}#/detectors/{detector}/comparisons/{index}")
        return pack

    def _add_record(self, title: str, record: dict, kind: str, origin: str) -> None:
        sid = f"S{max((int(key[1:]) for key in self.sources), default=0) + 1}"
        leaves = list(_record_leaves(record))
        body = "\n".join(f"{key}: {value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)}"
                         for key, value in leaves)
        metadata = {"kind": kind, "origin": origin}
        detector = re.search(r"/detectors/([^/]+)", origin)
        if detector:
            metadata["detector"] = detector.group(1)
        batches = [record[key] for key in ("batch", "lot_a", "lot_b") if isinstance(record.get(key), str)]
        if batches:
            metadata["batches"] = ",".join(batches)
        self.sources[sid] = Source(sid, title, body, metadata, dict(leaves))
        for key, value in leaves:
            if kind == "reference" and isinstance(value, str) and not re.search(r"\d", value):
                continue
            if ".supporting_references." in f".{key}.":
                continue
            fid = f"context.{sid}.{key}"
            self.context_facts[fid] = Fact(
                id=fid, kind="context", value=value, display=_context_display(value, key, kind),
                label=f"{title}: {key}", note=f"[{sid}] {origin} / {key}; original value: {value}",
            )

    def with_sample(self, sample: dict, *, origin: str = "sample JSON") -> "KnowledgePack":
        if not isinstance(sample, dict) or not sample:
            raise ValueError("sample JSON must be a non-empty object")
        raw = self.raw + "\n" + json.dumps(sample, ensure_ascii=False, allow_nan=False)
        pack = KnowledgePack(dict(self.sources), raw=raw, path=self.path, context_facts=self.context_facts)
        pack._add_record("Supplied sample output (native upstream schema)", sample, "sample", origin)
        return pack

    def select(self, query: str, *, limit: int = 6, compact: bool = True,
               batch_statistics: bool = True) -> "KnowledgePack":
        aliases = {"porosity": "pore", "binder": "CBD", "siox": "silicon", "crack": "fracture"}
        query = query + " " + " ".join(value for key, value in aliases.items() if key in query.lower())
        terms = set(re.findall(r"[a-z0-9]+", query.lower())) - {"the", "and", "for", "why", "how", "what", "with"}
        ranked = []
        for sid, source in self.sources.items():
            if source.metadata.get("kind") != "reference":
                continue
            title = source.title.lower()
            body = "\n".join(str(value) for key, value in source.fields.items()
                             if not key.startswith("supporting_references.")) if source.fields else source.body
            body = body.lower()
            score = sum(4 * (term in title) + (term in body) for term in terms if len(term) > 2)
            ranked.append((score, sid))
        wanted = {sid for _, sid in sorted(ranked, key=lambda item: (-item[0], int(item[1][1:])))[:limit]}
        detectors = set(re.findall(r"\b(?:bse|etd|inlens)\b", query.lower()))
        batches = {f"batch_{number}" for number in re.findall(r"\bbatch[_ -]?(\d+)\b", query.lower())}
        sources = {}
        for sid, source in self.sources.items():
            kind = source.metadata.get("kind")
            if kind == "reference" and sid not in wanted:
                continue
            if kind in {"batch", "comparison"}:
                if not batch_statistics and source.metadata.get("batches"):
                    continue
                if detectors and source.metadata.get("detector") not in detectors:
                    continue
                source_batches = set(source.metadata.get("batches", "").lower().split(",")) - {""}
                if batches and source_batches:
                    if len(batches) > 1 and not source_batches <= batches:
                        continue
                    if len(batches) == 1 and not source_batches & batches:
                        continue
            sources[sid] = _compact_source(source, query) if compact else source
        facts = {}
        for fid, fact in self.context_facts.items():
            _, sid, key = fid.split(".", 2)
            if sid not in sources:
                continue
            source = sources[sid]
            if source.fields is not None and key not in source.fields:
                continue
            if compact and source.metadata.get("kind") != "sample" and isinstance(fact.value, str) and not re.search(r"\d", fact.value):
                continue
            facts[fid] = fact
        return KnowledgePack(sources, raw=self.raw, path=self.path, context_facts=facts)

    @classmethod
    def empty(cls) -> "KnowledgePack":
        return cls({}, raw="")

    # -- properties ------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self.sources)

    def __contains__(self, source_id: object) -> bool:
        return source_id in self.sources

    @property
    def version(self) -> str:
        """Content hash, recorded in the audit trail so a pack edit is visible."""
        return "sha256:" + hashlib.sha256(self.raw.encode("utf-8")).hexdigest()[:16]

    def ids(self) -> list[str]:
        return sorted(self.sources, key=lambda s: int(s[1:]))

    # -- checking --------------------------------------------------------- #
    def verify_quote(self, source_id: str, quote: str) -> QuoteCheck:
        quote = (quote or "").strip()
        if not quote:
            return QuoteCheck(False, source_id, quote, reason="the quote is empty")
        source = self.sources.get(source_id)
        if source is None:
            return QuoteCheck(
                False, source_id, quote,
                reason=f"source {source_id} is not in the knowledge pack "
                       f"(available: {', '.join(self.ids()) or 'none'})",
            )
        if source.contains(quote):
            return QuoteCheck(True, source_id, quote)
        return QuoteCheck(
            False, source_id, quote,
            reason=f"this wording does not appear in {source_id}",
            suggestion=source.closest_sentence(quote),
        )

    def undefined_markers(self, text: str) -> list[str]:
        """``[S7]`` markers in prose that no source backs."""
        found = {f"S{m.group(1)}" for m in CITATION_MARKER.finditer(text)}
        return sorted(found - set(self.sources), key=lambda s: int(s[1:]))

    # -- v2 read-only tool ------------------------------------------------ #
    def search(self, query: str, *, limit: int = 3, window: int = 320) -> list[dict[str, str]]:
        """Keyword search returning verbatim snippets the model may quote."""
        terms = [t for t in re.findall(r"[A-Za-z0-9]+", query.lower()) if len(t) > 2]
        if not terms:
            return []
        hits: list[tuple[int, str, str]] = []
        for source in self.sources.values():
            text = source.body
            low = text.lower()
            score = sum(low.count(t) for t in terms)
            if not score:
                continue
            first = min((low.find(t) for t in terms if low.find(t) >= 0), default=0)
            start = max(0, first - window // 4)
            snippet = text[start : start + window].strip()
            hits.append((score, source.id, snippet))
        hits.sort(key=lambda h: -h[0])
        return [
            {"source": sid, "title": self.sources[sid].title, "snippet": snippet}
            for _, sid, snippet in hits[:limit]
        ]

    # -- prompt rendering ------------------------------------------------- #
    def to_prompt_block(self) -> str:
        if not self.sources:
            return "(The knowledge pack is empty. Make no materials-science claims.)"
        out: list[str] = []
        if any(source.metadata.get("context_view") == "compact" for source in self.sources.values()):
            out.append("Focused context: bibliography and unrequested auxiliary fields may be omitted. "
                       "Do not infer that omitted measurements or references do not exist.")
        for sid in self.ids():
            source = self.sources[sid]
            header = f"### [{sid}] {source.title}"
            if source.url:
                header += f"  <{source.url}>"
            if source.metadata.get("origin"):
                header += f"  (local source: {source.metadata['origin']})"
            out.append(header)
            out.append(source.body)
            out.append("")
        return "\n".join(out)


def _compact_source(source: Source, query: str) -> Source:
    if source.fields is None or source.metadata.get("kind") == "sample":
        return source
    terms = set(re.findall(r"[a-z0-9]+", query.lower()))
    bibliography = bool(terms & {"doi", "references", "papers", "bibliography", "publications"})
    phases = set()
    if terms & {"pore", "pores", "porosity"}:
        phases.add("pore")
    if "graphite" in terms:
        phases.add("graphite")
    if "bright" in terms:
        phases.add("bright phase")
    if re.search(r"\b(?:all|other) (?:phases|statistics|features)\b", query.lower()):
        phases.clear()
    phase_names = {key.rsplit(".", 1)[0]: value for key, value in source.fields.items()
                   if key.startswith("phases.") and key.endswith(".phase")}
    auxiliary = {"bits", "next", "variant_shifts", "interval90", "interval_width", "to_settle"}
    if terms & {"uncertainty", "budget", "width", "sensitivity", "segmentation", "next", "more", "imaging"}:
        auxiliary.clear()
    fields = {}
    for key, value in source.fields.items():
        parts = key.split(".")
        if parts[0] == "supporting_references" and not bibliography:
            continue
        if source.metadata.get("kind") in {"batch", "comparison"}:
            if phases and len(parts) >= 3 and parts[0] == "phases":
                phase = phase_names.get(".".join(parts[:2]), parts[1])
                if phase not in phases:
                    continue
            omitted = set(parts) & auxiliary
            if omitted and not any(set(re.findall(r"[a-z0-9]+", name)) <= terms for name in omitted):
                continue
        fields[key] = value
    body = "\n".join(f"{key}: {value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)}"
                     for key, value in fields.items())
    return Source(source.id, source.title, body, {**source.metadata, "context_view": "compact"}, fields)


def _record_leaves(value, prefix=""):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _record_leaves(item, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _record_leaves(item, f"{prefix}.{index}")
    else:
        yield prefix, value


def _context_display(value, key, kind):
    if isinstance(value, str):
        return value
    if value is None:
        return "not available"
    if isinstance(value, bool):
        return "yes" if value else "no"
    parts = key.split(".")
    field = parts[-2] if parts[-1].isdigit() and len(parts) > 1 else parts[-1]
    if kind in {"batch", "comparison"}:
        if field in {"delta", "se_sampling", "systematic_halfwidth"} or (
            kind == "comparison" and field in {"interval95", "interval90"}
        ):
            return f"{value * 100:+.2f} percentage points"
        if field in {"phi", "phi_a", "phi_b", "ci95", "prediction_interval_new_field", "threshold_range",
                     "systematic_range", "tolerance", "I2", "p_direction", "p_beyond_tolerance"}:
            return f"{value * 100:.2f}%"
        if field == "delta_rel":
            return f"{value * 100:+.2f}%"
        if field == "target_pixel_nm":
            return f"{value:.6g} nm per pixel"
    if kind == "sample" and key.startswith("probabilities."):
        return f"{value * 100:.1f}%"
    if kind == "sample" and (key.startswith("phase_fractions_pct.") or field.endswith("_pct")):
        return f"{value:.6g}%"
    return str(value) if isinstance(value, int) else f"{value:.6g}"
