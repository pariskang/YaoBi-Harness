"""Licence model for external knowledge sources.

The harness ships **code, not content**. Every source declares how its material
may be reused, and the knowledge store refuses writes that would violate that
declaration. This is enforcement, not documentation: a CC BY-NC-SA dataset
cannot be ingested into a commercial deployment, a read-only source cannot have
its full text stored, and a licensed source stays disabled until the operator
records an attestation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Reuse(str, Enum):
    """How far a source's content may be reused."""

    #: CC0 / public domain — copy, modify, redistribute, commercial use.
    PUBLIC_DOMAIN = "public_domain"
    #: CC BY or equivalent — reuse with attribution, commercial allowed.
    OPEN_ATTRIBUTION = "open_attribution"
    #: CC BY-NC-SA and friends — non-commercial only, share-alike.
    NONCOMMERCIAL = "noncommercial"
    #: Free to read, no redistribution right. Metadata + citation + URL only.
    LINK_ONLY = "link_only"
    #: Requires a purchased or granted licence before any use.
    CREDENTIALED = "credentialed"


class DeploymentMode(str, Enum):
    RESEARCH = "research_noncommercial"
    COMMERCIAL = "commercial"


class LicenseError(RuntimeError):
    """Raised when an operation would exceed a source's licence."""


@dataclass(frozen=True)
class SourceLicense:
    license_name: str
    reuse: Reuse
    attribution: str = ""
    share_alike: bool = False
    #: Human-readable statement shown alongside any answer citing this source.
    notice: str = ""
    url: str = ""

    @property
    def allows_full_text_storage(self) -> bool:
        return self.reuse in (Reuse.PUBLIC_DOMAIN, Reuse.OPEN_ATTRIBUTION, Reuse.NONCOMMERCIAL, Reuse.CREDENTIALED)

    @property
    def allows_commercial(self) -> bool:
        return self.reuse in (Reuse.PUBLIC_DOMAIN, Reuse.OPEN_ATTRIBUTION, Reuse.CREDENTIALED)


@dataclass(frozen=True)
class Attestation:
    """Operator's record that a licence for a credentialed source is held."""

    licensee: str
    license_reference: str
    expires: str = ""
    scope: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "licensee": self.licensee,
            "license_reference": self.license_reference,
            "expires": self.expires,
            "scope": self.scope,
        }


@dataclass
class LicensePolicy:
    """Decides, per source, what this deployment is permitted to do.

    ``attestations`` maps ``source_id`` to an :class:`Attestation`; a
    credentialed source without one stays disabled. Attestations can also be
    supplied out-of-band via ``YAOBI_LICENSE_ATTESTATIONS`` (a JSON object).
    """

    mode: DeploymentMode = DeploymentMode.RESEARCH
    attestations: dict[str, Attestation] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "LicensePolicy":
        import json

        raw_mode = (os.environ.get("YAOBI_DEPLOYMENT_MODE") or DeploymentMode.RESEARCH.value).strip().lower()
        try:
            mode = DeploymentMode(raw_mode)
        except ValueError as exc:
            raise LicenseError(
                f"unknown YAOBI_DEPLOYMENT_MODE {raw_mode!r}; expected "
                f"{DeploymentMode.RESEARCH.value} or {DeploymentMode.COMMERCIAL.value}"
            ) from exc

        attestations: dict[str, Attestation] = {}
        blob = os.environ.get("YAOBI_LICENSE_ATTESTATIONS")
        if blob:
            try:
                parsed = json.loads(blob)
            except ValueError as exc:
                raise LicenseError(f"YAOBI_LICENSE_ATTESTATIONS is not valid JSON: {exc}") from exc
            for source_id, entry in (parsed or {}).items():
                if not isinstance(entry, dict) or not entry.get("licensee") or not entry.get("license_reference"):
                    raise LicenseError(f"attestation for {source_id!r} needs 'licensee' and 'license_reference'")
                attestations[source_id] = Attestation(
                    licensee=str(entry["licensee"]),
                    license_reference=str(entry["license_reference"]),
                    expires=str(entry.get("expires", "")),
                    scope=str(entry.get("scope", "")),
                )
        return cls(mode=mode, attestations=attestations)

    # ------------------------------------------------------------------ checks
    def evaluate(self, source: Any) -> tuple[bool, str]:
        """Return ``(enabled, reason)`` for a :class:`~.sources.KnowledgeSource`."""
        lic = source.license
        if lic.reuse is Reuse.CREDENTIALED and source.source_id not in self.attestations:
            return False, "credentialed_source_requires_license_attestation"
        if self.mode is DeploymentMode.COMMERCIAL and not lic.allows_commercial:
            return False, f"license_{lic.reuse.value}_forbids_commercial_deployment"
        return True, "enabled"

    def require(self, source: Any) -> None:
        enabled, reason = self.evaluate(source)
        if not enabled:
            raise LicenseError(f"{source.source_id}: {reason} ({source.license.license_name})")

    def may_store_full_text(self, source: Any) -> bool:
        """Read-only sources may keep citations and metadata, never body text."""
        return source.license.allows_full_text_storage and self.evaluate(source)[0]

    def notices_for(self, source_ids: list[str], catalog: dict[str, Any]) -> list[dict[str, str]]:
        """Attribution/share-alike notices that must accompany an answer."""
        notices = []
        for source_id in dict.fromkeys(source_ids):
            source = catalog.get(source_id)
            if source is None:
                continue
            notices.append(
                {
                    "source_id": source_id,
                    "name": source.name,
                    "license": source.license.license_name,
                    "attribution": source.license.attribution,
                    "notice": source.license.notice,
                    "share_alike": "yes" if source.license.share_alike else "no",
                    "url": source.license.url,
                }
            )
        return notices

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "attested_sources": sorted(self.attestations),
        }
