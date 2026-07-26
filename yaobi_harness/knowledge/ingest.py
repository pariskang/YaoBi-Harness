"""Operator-run knowledge build.

Nothing here runs during a clinical run. An operator executes it once (and then
on a refresh schedule) to populate the local store from sources their
deployment is licensed to use:

    python -m yaobi_harness knowledge sources
    python -m yaobi_harness knowledge build --store ./knowledge.db
    python -m yaobi_harness knowledge ingest-file --store ./knowledge.db \
        --source chp_2025 --kind dose_ranges --path ./authorized/chp2025_herbs.json
    python -m yaobi_harness knowledge stats --store ./knowledge.db
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .connectors.base import ConnectorError
from .connectors.files import FILE_LOADERS, FileIngestError, ingest_ddinter, load_builtin_rule_packs
from .connectors.web import WEB_CONNECTORS
from .licensing import LicenseError, LicensePolicy
from .sources import SOURCE_CATALOG, catalog_summary
from .store import KnowledgeStore

#: Ingredients worth having a current label for in an orthopaedic setting.
#: These are exactly the drugs the built-in interaction rules reason about.
ORTHOPAEDIC_INGREDIENTS = (
    "ibuprofen", "naproxen", "diclofenac sodium", "celecoxib", "meloxicam", "indomethacin",
    "aspirin", "warfarin sodium", "rivaroxaban", "apixaban", "dabigatran etexilate",
    "enoxaparin sodium", "clopidogrel", "ticagrelor",
    "prednisone", "methylprednisolone", "dexamethasone",
    "tramadol hydrochloride", "oxycodone hydrochloride", "morphine sulfate", "fentanyl",
    "acetaminophen", "gabapentin", "pregabalin", "duloxetine hydrochloride",
    "alendronate sodium", "risedronate sodium", "zoledronic acid", "ibandronate sodium",
    "denosumab", "teriparatide", "romosozumab",
    "methotrexate", "colchicine", "cyclobenzaprine", "tizanidine",
)

#: Guideline topics to pull when a NICE syndication licence is configured.
ORTHOPAEDIC_TOPICS = (
    "low back pain and sciatica", "osteoarthritis", "hip fracture", "non-complex fractures",
    "spinal injury", "joint replacement", "venous thromboembolism prophylaxis",
    "osteoporosis fragility fracture", "chronic pain",
)


def open_store(path: str | Path | None = None, policy: LicensePolicy | None = None) -> KnowledgeStore:
    return KnowledgeStore(path, policy or LicensePolicy.from_env())


def list_sources(policy: LicensePolicy | None = None) -> list[dict[str, Any]]:
    """Catalogue plus whether this deployment may currently use each source."""
    active = policy or LicensePolicy.from_env()
    rows = []
    for entry in catalog_summary():
        source = SOURCE_CATALOG[entry["source_id"]]
        enabled, reason = active.evaluate(source)
        rows.append({**entry, "enabled": enabled, "reason": reason})
    return rows


def build(
    store: KnowledgeStore,
    *,
    ingredients: Iterable[str] = ORTHOPAEDIC_INGREDIENTS,
    topics: Iterable[str] = ORTHOPAEDIC_TOPICS,
    cache_dir: str | Path | None = None,
    nice_api_key: str | None = None,
    include: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Populate the store from every source this deployment is allowed to fetch.

    Sources the licence policy blocks are reported as skipped rather than
    raising, so a research build and a commercial build can share one command.
    """
    wanted = set(include) if include else None
    report: dict[str, Any] = {"built": [], "skipped": [], "errors": []}

    report["built"].append(load_builtin_rule_packs(store))

    for source_id, connector_cls in WEB_CONNECTORS.items():
        if wanted and source_id not in wanted:
            continue
        source = SOURCE_CATALOG[source_id]
        enabled, reason = store.policy.evaluate(source)
        if not enabled:
            report["skipped"].append({"source": source_id, "reason": reason, "license": source.license.license_name})
            continue
        try:
            connector = connector_cls(
                store.policy,
                cache_dir=cache_dir,
                api_key=nice_api_key if source_id == "nice" else None,
            )
        except LicenseError as exc:
            report["skipped"].append({"source": source_id, "reason": str(exc)})
            continue

        try:
            if source_id in ("openfda", "dailymed"):
                report["built"].append(connector.ingest(store, ingredients))
            elif source_id == "nice":
                report["built"].append(connector.ingest(store, topics))
            elif source_id == "rxnorm":
                # RxNorm is queried live for name resolution; nothing to persist.
                report["built"].append({"source": "rxnorm", "mode": "live_lookup_only"})
        except (ConnectorError, LicenseError) as exc:
            report["errors"].append({"source": source_id, "error": str(exc)})

    report["stats"] = store.stats()
    return report


def ingest_file(store: KnowledgeStore, source_id: str, kind: str, path: str | Path, version: str = "") -> dict[str, Any]:
    """Load an operator-supplied export into the store."""
    if source_id == "ddinter":
        return ingest_ddinter(store, path, version=version or "2024-05-14")
    loader = FILE_LOADERS.get(kind)
    if loader is None:
        raise FileIngestError(f"unknown ingest kind {kind!r}; expected one of {sorted(FILE_LOADERS)}")
    if kind == "citations":
        return loader(store, source_id, path)
    return loader(store, source_id, path, version=version)
