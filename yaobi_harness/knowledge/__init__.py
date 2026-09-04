"""External knowledge: guidelines, drug labels, dose ranges and interactions.

The repository ships connectors and a licence model, never third-party content.
See :mod:`yaobi_harness.knowledge.sources` for the catalogue and
:mod:`yaobi_harness.knowledge.ingest` for the operator-run build step.
"""

from .licensing import Attestation, DeploymentMode, LicenseError, LicensePolicy, Reuse, SourceLicense
from .ortho_interactions import ORTHO_RULES, evaluate as evaluate_drug_interactions, rule_pack_summary
from .sources import DEFAULT_ENABLED, SOURCE_CATALOG, catalog_summary, get_source
from .store import KnowledgeStore, Provenance

__all__ = [
    "Attestation", "DeploymentMode", "LicenseError", "LicensePolicy", "Reuse", "SourceLicense",
    "ORTHO_RULES", "evaluate_drug_interactions", "rule_pack_summary",
    "DEFAULT_ENABLED", "SOURCE_CATALOG", "catalog_summary", "get_source",
    "KnowledgeStore", "Provenance",
]
