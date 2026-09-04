"""Connectors for sources with a public or credentialed HTTP API.

* **openFDA** (CC0) — structured US drug labels: interactions, contraindications,
  dosing, special populations. The open backbone of the medication layer.
* **DailyMed** (public) — the same labels as SPL, used to confirm the current
  version of a label set.
* **RxNorm / RxClass** (public) — normalises brand/generic/salt/strength names
  and returns ATC classes, so a free-text medication list becomes matchable.
* **NICE** (credentialed) — the syndication API; stays disabled until an
  attestation is recorded, because access requires an application to NICE.
"""

from __future__ import annotations

from typing import Any, Iterable

from ..store import LABEL_SECTIONS, KnowledgeStore
from .base import ConnectorError, HttpConnector


class OpenFdaConnector(HttpConnector):
    """openFDA drug label API."""

    source_id = "openfda"

    def fetch_label(self, ingredient: str, limit: int = 1) -> list[dict[str, Any]]:
        query = f'openfda.generic_name:"{ingredient}"'
        payload = self.get_json("", {"search": query, "limit": limit})
        if not payload:
            # Fall back to brand name before giving up.
            payload = self.get_json("", {"search": f'openfda.brand_name:"{ingredient}"', "limit": limit})
        return (payload or {}).get("results", []) or []

    def ingest(self, store: KnowledgeStore, ingredients: Iterable[str], sections: Iterable[str] = LABEL_SECTIONS) -> dict[str, Any]:
        """Store the clinically relevant sections of each ingredient's label."""
        wanted = list(sections)
        stored, missing = 0, []
        for ingredient in ingredients:
            try:
                results = self.fetch_label(ingredient)
            except ConnectorError:
                missing.append(ingredient)
                continue
            if not results:
                missing.append(ingredient)
                continue
            record = results[0]
            openfda = record.get("openfda", {}) or {}
            rxcui = ",".join(openfda.get("rxcui", [])[:5])
            brand = (openfda.get("brand_name") or [""])[0]
            for section in wanted:
                text = record.get(section)
                if not text:
                    continue
                store.add_label_section(
                    self.source_id,
                    ingredient=ingredient,
                    section=section,
                    text=" ".join(text) if isinstance(text, list) else str(text),
                    brand=brand,
                    rxcui=rxcui,
                    set_id=str(record.get("set_id", "")),
                    label_version=str(record.get("version", "")),
                    effective_time=str(record.get("effective_time", "")),
                    url=f"https://api.fda.gov/drug/label.json?search=set_id:{record.get('set_id', '')}",
                )
                stored += 1
        return {"source": self.source_id, "sections_stored": stored, "not_found": missing}


class DailyMedConnector(HttpConnector):
    """DailyMed SPL service — used to resolve and version a label set."""

    source_id = "dailymed"

    def find_spl(self, drug_name: str, pagesize: int = 5) -> list[dict[str, Any]]:
        payload = self.get_json("spls.json", {"drug_name": drug_name, "pagesize": pagesize})
        return (payload or {}).get("data", []) or []

    def ingest(self, store: KnowledgeStore, ingredients: Iterable[str]) -> dict[str, Any]:
        stored, missing = 0, []
        for ingredient in ingredients:
            try:
                entries = self.find_spl(ingredient, pagesize=1)
            except ConnectorError:
                missing.append(ingredient)
                continue
            if not entries:
                missing.append(ingredient)
                continue
            entry = entries[0]
            set_id = str(entry.get("setid", ""))
            store.add_label_section(
                self.source_id,
                ingredient=ingredient,
                section="label_reference",
                text=str(entry.get("title", "")),
                set_id=set_id,
                label_version=str(entry.get("spl_version", "")),
                effective_time=str(entry.get("published_date", "")),
                url=f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={set_id}",
            )
            stored += 1
        return {"source": self.source_id, "labels_stored": stored, "not_found": missing}


class RxNormConnector(HttpConnector):
    """RxNorm / RxClass name normalisation and ATC classification."""

    source_id = "rxnorm"

    def rxcui_for(self, name: str) -> str | None:
        payload = self.get_json("rxcui.json", {"name": name, "search": 2})
        ids = ((payload or {}).get("idGroup", {}) or {}).get("rxnormId") or []
        return str(ids[0]) if ids else None

    def normalized_name(self, rxcui: str) -> str | None:
        payload = self.get_json(f"rxcui/{rxcui}/property.json", {"propName": "RxNormName"})
        group = (payload or {}).get("propConceptGroup", {}) or {}
        concepts = group.get("propConcept") or []
        return concepts[0].get("propValue") if concepts else None

    def atc_classes(self, name: str) -> list[dict[str, str]]:
        payload = self.get_json("rxclass/class/byDrugName.json", {"drugName": name, "relaSource": "ATC"})
        entries = ((payload or {}).get("rxclassDrugInfoList", {}) or {}).get("rxclassDrugInfo") or []
        seen, out = set(), []
        for entry in entries:
            item = entry.get("rxclassMinConceptItem", {}) or {}
            class_id = item.get("classId")
            if class_id and class_id not in seen:
                seen.add(class_id)
                out.append({"atc_code": class_id, "atc_name": item.get("className", "")})
        return out

    def normalize(self, name: str) -> dict[str, Any]:
        """Best-effort identity resolution for one free-text medication name."""
        result: dict[str, Any] = {"input": name, "rxcui": None, "normalized_name": None, "atc": []}
        try:
            rxcui = self.rxcui_for(name)
            result["rxcui"] = rxcui
            if rxcui:
                result["normalized_name"] = self.normalized_name(rxcui)
            result["atc"] = self.atc_classes(name)
        except ConnectorError as exc:
            result["error"] = str(exc)
        return result


class NiceConnector(HttpConnector):
    """NICE syndication API.

    Disabled until an attestation for ``nice`` is recorded, because NICE grants
    access by application and the terms vary by territory and use.
    """

    source_id = "nice"

    def headers(self) -> dict[str, str]:
        headers = super().headers()
        if self.api_key:
            headers["API-Key"] = self.api_key
        return headers

    def search(self, topic: str, limit: int = 10) -> list[dict[str, Any]]:
        payload = self.get_json("search", {"q": topic, "pageSize": limit})
        if not payload:
            return []
        for key in ("results", "documents", "items"):
            if isinstance(payload.get(key), list):
                return payload[key]
        return []

    def ingest(self, store: KnowledgeStore, topics: Iterable[str], limit: int = 10) -> dict[str, Any]:
        stored = 0
        for topic in topics:
            for entry in self.search(topic, limit=limit):
                identifier = str(entry.get("id") or entry.get("reference") or entry.get("url") or "")
                if not identifier:
                    continue
                store.add_guideline(
                    self.source_id,
                    guideline_id=identifier,
                    title=str(entry.get("title", identifier)),
                    topic=topic,
                    url=str(entry.get("url", "")),
                    published=str(entry.get("publicationDate") or entry.get("published", "")),
                    version=str(entry.get("version", "")),
                    evidence_grade=str(entry.get("evidenceGrade", "")),
                    summary=str(entry.get("summary") or entry.get("description", "")),
                    recommendations=[str(r) for r in (entry.get("recommendations") or [])],
                )
                stored += 1
        return {"source": self.source_id, "guidelines_stored": stored}


WEB_CONNECTORS = {
    "openfda": OpenFdaConnector,
    "dailymed": DailyMedConnector,
    "rxnorm": RxNormConnector,
    "nice": NiceConnector,
}
