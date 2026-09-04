"""Clinical image reading: non-diagnostic by construction.

A read can raise a red flag, suggest what to ask and describe what is visible.
It can never diagnose, never replace a formal report, and never survive a PHI
pre-check that found identifiers. See :mod:`.client`.
"""

from .client import (
    IMAGE_KINDS, MAX_IMAGE_BYTES, ImageRead, VisionClient, VisionError,
    build_vision_client, decode_data_uri, describe_vision, encode_image,
)

__all__ = [
    "IMAGE_KINDS", "MAX_IMAGE_BYTES", "ImageRead", "VisionClient", "VisionError",
    "build_vision_client", "decode_data_uri", "describe_vision", "encode_image",
]
