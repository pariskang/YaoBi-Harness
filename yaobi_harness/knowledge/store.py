"""Local knowledge store (SQLite), with licence enforcement on every write.

The repository ships an **empty** store. Operators populate it with
:mod:`yaobi_harness.knowledge.ingest`, and the store refuses any write that
would exceed the source's declared reuse terms:

* a non-commercial source cannot be ingested into a commercial deployment;
* a read-only source can hold citations and metadata but never body text;
* a credentialed source stays closed until an attestation is recorded.

Everything retrieved carries provenance — source, version, publication date and
retrieval time — so answers can state where each fact came from.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .licensing import LicenseError, LicensePolicy, Reuse
from .sources import SOURCE_CATALOG, KnowledgeSource, get_source

SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
    source_id     TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    license       TEXT NOT NULL,
    reuse         TEXT NOT NULL,
    version       TEXT DEFAULT '',
    retrieved_at  TEXT DEFAULT '',
    url           TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS guidelines (
    guideline_id   TEXT PRIMARY KEY,
    source_id      TEXT NOT NULL,
    title          TEXT NOT NULL,
    topic          TEXT DEFAULT '',
    url            TEXT DEFAULT '',
    published      TEXT DEFAULT '',
    version        TEXT DEFAULT '',
    evidence_grade TEXT DEFAULT '',
    summary        TEXT DEFAULT '',
    body           TEXT DEFAULT '',
    recommendations TEXT DEFAULT '[]',
    retrieved_at   TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS label_sections (
    label_key      TEXT PRIMARY KEY,
    source_id      TEXT NOT NULL,
    ingredient     TEXT NOT NULL,
    brand          TEXT DEFAULT '',
    rxcui          TEXT DEFAULT '',
    set_id         TEXT DEFAULT '',
    label_version  TEXT DEFAULT '',
    effective_time TEXT DEFAULT '',
    section        TEXT NOT NULL,
    text           TEXT DEFAULT '',
    url            TEXT DEFAULT '',
    retrieved_at   TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS dose_ranges (
    dose_key       TEXT PRIMARY KEY,
    source_id      TEXT NOT NULL,
    substance      TEXT NOT NULL,
    substance_type TEXT NOT NULL DEFAULT 'herb',
    population     TEXT DEFAULT 'adult',
    route          TEXT DEFAULT 'oral',
    min_value      REAL,
    max_value      REAL,
    unit           TEXT DEFAULT 'g',
    basis          TEXT DEFAULT '',
    version        TEXT DEFAULT '',
    retrieved_at   TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS interactions (
    interaction_key TEXT PRIMARY KEY,
    source_id       TEXT NOT NULL,
    subject         TEXT NOT NULL,
    object          TEXT NOT NULL,
    severity        TEXT NOT NULL,
    mechanism       TEXT DEFAULT '',
    management      TEXT DEFAULT '',
    evidence        TEXT DEFAULT '',
    version         TEXT DEFAULT '',
    retrieved_at    TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS ix_guidelines_topic ON guidelines(topic);
CREATE INDEX IF NOT EXISTS ix_labels_ingredient ON label_sections(ingredient);
CREATE INDEX IF NOT EXISTS ix_dose_substance ON dose_ranges(substance);
CREATE INDEX IF NOT EXISTS ix_interactions_subject ON interactions(subject);
CREATE INDEX IF NOT EXISTS ix_interactions_object ON interactions(object);
"""

#: Label sections worth keeping for a clinical decision-support system.
LABEL_SECTIONS = (
    "drug_interactions",
    "contraindications",
    "warnings_and_precautions",
    "warnings",
    "dosage_and_administration",
    "use_in_specific_populations",
    "adverse_reactions",
    "boxed_warning",
)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass
class Provenance:
    source_id: str
    source_name: str
    license: str
    version: str = ""
    published: str = ""
    retrieved_at: str = ""
    url: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "source_id": self.source_id,
            "source": self.source_name,
            "license": self.license,
            "version": self.version,
            "published": self.published,
            "retrieved_at": self.retrieved_at,
            "url": self.url,
        }


class KnowledgeStore:
    """SQLite-backed store for guidelines, labels, dose ranges and interactions."""

    def __init__(self, path: str | Path | None = None, policy: LicensePolicy | None = None) -> None:
        self.path = Path(path) if path else Path(":memory:")
        self.policy = policy or LicensePolicy()
        if self.path != Path(":memory:"):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "KnowledgeStore":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -------------------------------------------------------------- licensing
    def _authorize(self, source_id: str) -> KnowledgeSource:
        source = get_source(source_id)
        self.policy.require(source)
        return source

    def register_source(self, source_id: str, version: str = "", url: str = "") -> None:
        source = self._authorize(source_id)
        self.conn.execute(
            "INSERT OR REPLACE INTO sources(source_id, name, license, reuse, version, retrieved_at, url)"
            " VALUES (?,?,?,?,?,?,?)",
            (source_id, source.name, source.license.license_name, source.license.reuse.value,
             version, now(), url or source.license.url),
        )
        self.conn.commit()

    def _text_or_empty(self, source: KnowledgeSource, text: str) -> str:
        """Strip body text for sources that grant no redistribution right."""
        return text if self.policy.may_store_full_text(source) else ""

    # ----------------------------------------------------------------- writes
    def add_guideline(
        self,
        source_id: str,
        guideline_id: str,
        title: str,
        *,
        topic: str = "",
        url: str = "",
        published: str = "",
        version: str = "",
        evidence_grade: str = "",
        summary: str = "",
        body: str = "",
        recommendations: Iterable[str] = (),
    ) -> None:
        source = self._authorize(source_id)
        if source.license.reuse is Reuse.LINK_ONLY and not url:
            raise LicenseError(f"{source_id}: read-only source requires a citation URL")
        self.register_source(source_id, version=version, url=url)
        self.conn.execute(
            "INSERT OR REPLACE INTO guidelines(guideline_id, source_id, title, topic, url, published, version,"
            " evidence_grade, summary, body, recommendations, retrieved_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                f"{source_id}:{guideline_id}", source_id, title, topic, url, published, version, evidence_grade,
                summary, self._text_or_empty(source, body),
                json.dumps(list(recommendations), ensure_ascii=False), now(),
            ),
        )
        self.conn.commit()

    def add_label_section(
        self,
        source_id: str,
        ingredient: str,
        section: str,
        text: str,
        *,
        brand: str = "",
        rxcui: str = "",
        set_id: str = "",
        label_version: str = "",
        effective_time: str = "",
        url: str = "",
    ) -> None:
        source = self._authorize(source_id)
        self.register_source(source_id, version=label_version, url=url)
        key = f"{source_id}:{ingredient.lower()}:{set_id or brand}:{section}"
        self.conn.execute(
            "INSERT OR REPLACE INTO label_sections(label_key, source_id, ingredient, brand, rxcui, set_id,"
            " label_version, effective_time, section, text, url, retrieved_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (key, source_id, ingredient.lower(), brand, rxcui, set_id, label_version, effective_time,
             section, self._text_or_empty(source, text), url, now()),
        )
        self.conn.commit()

    def add_dose_range(
        self,
        source_id: str,
        substance: str,
        min_value: float,
        max_value: float,
        *,
        substance_type: str = "herb",
        unit: str = "g",
        population: str = "adult",
        route: str = "oral",
        basis: str = "",
        version: str = "",
    ) -> None:
        self._authorize(source_id)
        if min_value is None or max_value is None or min_value > max_value:
            raise ValueError(f"invalid dose range for {substance}: {min_value}-{max_value}")
        self.register_source(source_id, version=version)
        key = f"{source_id}:{substance}:{population}:{route}:{unit}"
        self.conn.execute(
            "INSERT OR REPLACE INTO dose_ranges(dose_key, source_id, substance, substance_type, population, route,"
            " min_value, max_value, unit, basis, version, retrieved_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (key, source_id, substance, substance_type, population, route,
             float(min_value), float(max_value), unit, basis, version, now()),
        )
        self.conn.commit()

    def add_interaction(
        self,
        source_id: str,
        subject: str,
        object_: str,
        severity: str,
        *,
        mechanism: str = "",
        management: str = "",
        evidence: str = "",
        version: str = "",
    ) -> None:
        self._authorize(source_id)
        self.register_source(source_id, version=version)
        left, right = sorted([subject.strip().lower(), object_.strip().lower()])
        key = f"{source_id}:{left}|{right}"
        self.conn.execute(
            "INSERT OR REPLACE INTO interactions(interaction_key, source_id, subject, object, severity,"
            " mechanism, management, evidence, version, retrieved_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (key, source_id, left, right, severity.lower(), mechanism, management, evidence, version, now()),
        )
        self.conn.commit()

    # ------------------------------------------------------------------ reads
    def _provenance(self, row: sqlite3.Row) -> dict[str, str]:
        source = SOURCE_CATALOG.get(row["source_id"])
        return Provenance(
            source_id=row["source_id"],
            source_name=source.name if source else row["source_id"],
            license=source.license.license_name if source else "unknown",
            version=row["version"] if "version" in row.keys() else "",
            published=row["published"] if "published" in row.keys() else "",
            retrieved_at=row["retrieved_at"] if "retrieved_at" in row.keys() else "",
            url=row["url"] if "url" in row.keys() else "",
        ).to_dict()

    def enabled_sources(self) -> list[str]:
        """Sources present in the store that this deployment may still use."""
        rows = self.conn.execute("SELECT source_id FROM sources").fetchall()
        out = []
        for row in rows:
            source = SOURCE_CATALOG.get(row["source_id"])
            if source and self.policy.evaluate(source)[0]:
                out.append(row["source_id"])
        return sorted(out)

    def search_guidelines(self, topic: str, limit: int = 5) -> list[dict[str, Any]]:
        enabled = self.enabled_sources()
        if not enabled:
            return []
        terms = [t for t in _terms(topic) if t]
        rows = self.conn.execute(
            f"SELECT * FROM guidelines WHERE source_id IN ({_marks(enabled)})", enabled
        ).fetchall()
        scored = []
        for row in rows:
            haystack = " ".join([row["title"], row["topic"], row["summary"], row["recommendations"]]).lower()
            score = sum(haystack.count(term.lower()) for term in terms)
            if score:
                scored.append((score, row))
        results = []
        for _, row in sorted(scored, key=lambda x: -x[0])[:limit]:
            results.append({
                "guideline_id": row["guideline_id"],
                "title": row["title"],
                "topic": row["topic"],
                "evidence_grade": row["evidence_grade"],
                "summary": row["summary"],
                "recommendations": json.loads(row["recommendations"] or "[]"),
                "has_full_text": bool(row["body"]),
                "provenance": self._provenance(row),
            })
        return results

    def label_sections(self, ingredient: str, sections: Iterable[str] = LABEL_SECTIONS) -> list[dict[str, Any]]:
        enabled = self.enabled_sources()
        if not enabled:
            return []
        wanted = list(sections)
        rows = self.conn.execute(
            f"SELECT * FROM label_sections WHERE ingredient = ? AND source_id IN ({_marks(enabled)})"
            f" AND section IN ({_marks(wanted)})",
            [ingredient.strip().lower(), *enabled, *wanted],
        ).fetchall()
        return [
            {
                "ingredient": row["ingredient"],
                "brand": row["brand"],
                "section": row["section"],
                "text": row["text"],
                "label_version": row["label_version"],
                "effective_time": row["effective_time"],
                "provenance": {
                    **self._provenance(row),
                    "version": row["label_version"],
                    "published": row["effective_time"],
                },
            }
            for row in rows
        ]

    def dose_range(self, substance: str, population: str = "adult") -> dict[str, Any] | None:
        """Return the most authoritative range available for ``substance``."""
        enabled = self.enabled_sources()
        if not enabled:
            return None
        rows = self.conn.execute(
            f"SELECT * FROM dose_ranges WHERE substance = ? AND source_id IN ({_marks(enabled)})",
            [substance, *enabled],
        ).fetchall()
        if not rows:
            return None
        preferred = sorted(rows, key=lambda r: (
            0 if r["population"] == population else 1,
            _SOURCE_PRIORITY.get(r["source_id"], 99),
        ))[0]
        return {
            "substance": preferred["substance"],
            "min_value": preferred["min_value"],
            "max_value": preferred["max_value"],
            "unit": preferred["unit"],
            "population": preferred["population"],
            "route": preferred["route"],
            "basis": preferred["basis"],
            "provenance": self._provenance(preferred),
        }

    def interactions_for(self, names: Iterable[str]) -> list[dict[str, Any]]:
        enabled = self.enabled_sources()
        lowered = sorted({n.strip().lower() for n in names if n and n.strip()})
        if not enabled or len(lowered) < 2:
            return []
        rows = self.conn.execute(
            f"SELECT * FROM interactions WHERE source_id IN ({_marks(enabled)})"
            f" AND subject IN ({_marks(lowered)}) AND object IN ({_marks(lowered)})",
            [*enabled, *lowered, *lowered],
        ).fetchall()
        return [
            {
                "subject": row["subject"],
                "object": row["object"],
                "severity": row["severity"],
                "mechanism": row["mechanism"],
                "management": row["management"],
                "evidence": row["evidence"],
                "provenance": self._provenance(row),
            }
            for row in rows
        ]

    def stats(self) -> dict[str, Any]:
        counts = {}
        for table in ("sources", "guidelines", "label_sections", "dose_ranges", "interactions"):
            counts[table] = self.conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]
        return {
            "path": str(self.path),
            "counts": counts,
            "enabled_sources": self.enabled_sources(),
            "policy": self.policy.to_dict(),
        }


#: Lower number wins when several sources define the same dose range.
_SOURCE_PRIORITY = {
    "chp_2025": 0,
    "nmpa_labels": 1,
    "ph_eur": 2,
    "usp_nf": 3,
    "who_intl_pharmacopoeia": 4,
    "openfda": 5,
    "dailymed": 6,
}


def _marks(items: list[str]) -> str:
    return ",".join("?" for _ in items)


def _terms(text: str) -> list[str]:
    import re

    tokens = [t for t in re.split(r"\W+", text or "") if t]
    cjk = re.sub(r"[^一-鿿]", "", text or "")
    return tokens + [cjk[i : i + 2] for i in range(len(cjk) - 1)]
