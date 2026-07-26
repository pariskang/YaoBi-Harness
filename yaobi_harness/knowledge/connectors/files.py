"""Connectors for operator-supplied files.

Sources that cannot legally be fetched by this repository — a purchased
pharmacopoeia, an authorised society guideline, a licensed DDI export, a
non-commercial dataset download — are ingested from files the operator obtains
themselves. The licence gate still applies at write time, so a non-commercial
export cannot be loaded into a commercial deployment and a read-only source
keeps only its citation.

Expected file formats are plain JSON/CSV and documented per loader below, so
operators can map whatever export they hold onto them.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from ..licensing import LicenseError, LicensePolicy
from ..sources import Reuse, get_source
from ..store import KnowledgeStore


class FileIngestError(ValueError):
    """Raised when an operator-supplied file does not match the expected shape."""


def _load(path: str | Path) -> Any:
    file_path = Path(path)
    if not file_path.exists():
        raise FileIngestError(f"file not found: {file_path}")
    if file_path.suffix.lower() in (".csv", ".tsv"):
        delimiter = "\t" if file_path.suffix.lower() == ".tsv" else ","
        with file_path.open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle, delimiter=delimiter))
    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise FileIngestError(f"{file_path} is neither valid JSON nor CSV/TSV: {exc}") from exc


def _rows(payload: Any, key: str) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    if isinstance(payload, dict) and isinstance(payload.get(key), list):
        return [r for r in payload[key] if isinstance(r, dict)]
    raise FileIngestError(f"expected a list of objects, or an object with a {key!r} list")


def _pick(row: dict[str, Any], *names: str, default: str = "") -> str:
    for name in names:
        if row.get(name) not in (None, ""):
            return str(row[name])
    return default


# ------------------------------------------------------------------- loaders

def ingest_dose_ranges(store: KnowledgeStore, source_id: str, path: str | Path, *, version: str = "") -> dict[str, Any]:
    """Load authorised dose ranges.

    Row shape (JSON list or CSV columns)::

        {"substance": "独活", "min": 3, "max": 10, "unit": "g",
         "population": "adult", "route": "oral", "basis": "《中国药典》2025 一部"}

    This is how 《中国药典》 / NMPA / USP-NF ranges enter the system: the operator
    holds the licence, extracts the ranges, and the harness stores only numbers
    plus a citation — never the monograph text.
    """
    rows = _rows(_load(path), "dose_ranges")
    stored, skipped = 0, []
    for row in rows:
        substance = _pick(row, "substance", "herb", "drug", "name")
        low = _pick(row, "min", "min_value", "low", "min_g")
        high = _pick(row, "max", "max_value", "high", "max_g")
        if not substance or not low or not high:
            skipped.append(row)
            continue
        try:
            store.add_dose_range(
                source_id,
                substance=substance,
                min_value=float(low),
                max_value=float(high),
                substance_type=_pick(row, "substance_type", "type", default="herb"),
                unit=_pick(row, "unit", default="g"),
                population=_pick(row, "population", default="adult"),
                route=_pick(row, "route", default="oral"),
                basis=_pick(row, "basis", "reference", "monograph"),
                version=version or _pick(row, "version"),
            )
            stored += 1
        except (ValueError, TypeError):
            skipped.append(row)
    return {"source": source_id, "dose_ranges_stored": stored, "skipped": len(skipped)}


def ingest_interactions(store: KnowledgeStore, source_id: str, path: str | Path, *, version: str = "") -> dict[str, Any]:
    """Load a drug-drug interaction export.

    Row shape::

        {"subject": "warfarin", "object": "ibuprofen", "severity": "major",
         "mechanism": "...", "management": "...", "evidence": "label"}

    Used for DrugBank / BNF / Stockley's exports under licence, and for the
    DDInter download in a non-commercial deployment.
    """
    rows = _rows(_load(path), "interactions")
    stored, skipped = 0, []
    for row in rows:
        subject = _pick(row, "subject", "drug_a", "drugA", "drug1", "a")
        object_ = _pick(row, "object", "drug_b", "drugB", "drug2", "b")
        severity = _pick(row, "severity", "level", "risk", default="moderate").lower()
        if not subject or not object_:
            skipped.append(row)
            continue
        store.add_interaction(
            source_id, subject, object_, severity,
            mechanism=_pick(row, "mechanism", "description"),
            management=_pick(row, "management", "recommendation", "action"),
            evidence=_pick(row, "evidence", "evidence_level"),
            version=version or _pick(row, "version"),
        )
        stored += 1
    return {"source": source_id, "interactions_stored": stored, "skipped": len(skipped)}


def ingest_guidelines(store: KnowledgeStore, source_id: str, path: str | Path, *, version: str = "") -> dict[str, Any]:
    """Load guideline records.

    Row shape::

        {"id": "NG59", "title": "Low back pain and sciatica in over 16s",
         "topic": "low back pain", "url": "...", "published": "2020-12-11",
         "evidence_grade": "NICE", "summary": "...",
         "recommendations": ["...", "..."], "body": "optional full text"}

    For a ``link_only`` source (AAOS, 中华医学会, NMPA, 香港衞生署) the ``body``
    field is discarded by the store and a citation URL is mandatory.
    """
    rows = _rows(_load(path), "guidelines")
    source = get_source(source_id)
    stored, skipped = 0, []
    for row in rows:
        guideline_id = _pick(row, "id", "guideline_id", "reference")
        title = _pick(row, "title", "name")
        if not guideline_id or not title:
            skipped.append(row)
            continue
        recommendations = row.get("recommendations") or []
        if isinstance(recommendations, str):
            recommendations = [r.strip() for r in recommendations.split("|") if r.strip()]
        try:
            store.add_guideline(
                source_id,
                guideline_id=guideline_id,
                title=title,
                topic=_pick(row, "topic", "subject"),
                url=_pick(row, "url", "link"),
                published=_pick(row, "published", "date", "publication_date"),
                version=version or _pick(row, "version"),
                evidence_grade=_pick(row, "evidence_grade", "grade", "strength"),
                summary=_pick(row, "summary", "description"),
                body=_pick(row, "body", "full_text"),
                recommendations=[str(r) for r in recommendations],
            )
            stored += 1
        except LicenseError:
            raise
        except ValueError:
            skipped.append(row)
    return {
        "source": source_id,
        "guidelines_stored": stored,
        "skipped": len(skipped),
        "full_text_stored": source.license.reuse is not Reuse.LINK_ONLY,
    }


def ingest_ddinter(store: KnowledgeStore, path: str | Path, *, version: str = "2024-05-14") -> dict[str, Any]:
    """Load a DDInter 2.0 export.

    DDInter is CC BY-NC-SA 4.0, so this refuses to run in a commercial
    deployment — the licence gate in :class:`KnowledgeStore` raises before any
    row is written. Its columns are usually ``Drug_A``/``Drug_B``/``Level``.
    """
    policy: LicensePolicy = store.policy
    policy.require(get_source("ddinter"))
    rows = _rows(_load(path), "interactions")
    level_map = {"major": "major", "moderate": "moderate", "minor": "minor", "unknown": "moderate"}
    stored = 0
    for row in rows:
        subject = _pick(row, "Drug_A", "drug_a", "subject")
        object_ = _pick(row, "Drug_B", "drug_b", "object")
        if not subject or not object_:
            continue
        level = _pick(row, "Level", "level", "severity", default="moderate").strip().lower()
        store.add_interaction(
            "ddinter", subject, object_, level_map.get(level, "moderate"),
            mechanism=_pick(row, "Mechanism", "mechanism"),
            management=_pick(row, "Management", "management", "recommendation"),
            evidence="DDInter 2.0 pharmacist-curated",
            version=version,
        )
        stored += 1
    return {"source": "ddinter", "interactions_stored": stored, "version": version,
            "notice": "CC BY-NC-SA 4.0 — 非商业使用；须与最新说明书交叉校验"}


def ingest_citations(store: KnowledgeStore, source_id: str, path: str | Path) -> dict[str, Any]:
    """Load citation-only records for a read-only source.

    Stores title, version, publication date, URL and any manually extracted
    recommendation points — never redistributable full text.
    """
    source = get_source(source_id)
    if source.license.reuse is not Reuse.LINK_ONLY:
        raise FileIngestError(
            f"{source_id} is {source.license.reuse.value}; use ingest_guidelines for it instead"
        )
    return ingest_guidelines(store, source_id, path)


FILE_LOADERS = {
    "dose_ranges": ingest_dose_ranges,
    "interactions": ingest_interactions,
    "guidelines": ingest_guidelines,
    "citations": ingest_citations,
}


def load_builtin_rule_packs(store: KnowledgeStore) -> dict[str, Any]:
    """Register the in-repo rule packs so answers can cite them by source id."""
    from ..ortho_interactions import ORTHO_RULES

    store.register_source("yaobi_ortho_rules", version=f"rules-{len(ORTHO_RULES)}")
    store.register_source("yaobi_tcm_incompatibility", version="classic-18-19")
    return {"ortho_rules": len(ORTHO_RULES), "tcm_incompatibility": "registered"}


def ingest_ortho_reference_ranges(store: KnowledgeStore, path: str | Path | None = None) -> dict[str, Any]:
    """Convenience wrapper for the operator's authorised herb dose file."""
    if path is None:
        raise FileIngestError("授权剂量范围文件必须由部署方提供（例如《中国药典》2025 年版摘录）")
    return ingest_dose_ranges(store, "chp_2025", path, version="2025")


def sample_dose_range_file(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    """Write a template dose-range file so operators can see the expected shape."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"dose_ranges": list(rows)}, ensure_ascii=False, indent=2), encoding="utf-8")
    return out
