"""Explicit contracts between independently versioned platform services."""

from .contracts import (
    AdapterError,
    ImageIdentity,
    build_vlm_contracts,
    semantic_draft_to_template,
    xywh_to_xyxy,
    xyxy_to_normalized_xywh,
    xyxy_to_xywh,
)

__all__ = [
    "AdapterError",
    "ImageIdentity",
    "build_vlm_contracts",
    "semantic_draft_to_template",
    "xywh_to_xyxy",
    "xyxy_to_normalized_xywh",
    "xyxy_to_xywh",
]
