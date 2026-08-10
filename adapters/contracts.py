from __future__ import annotations

import hashlib
import io
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable

from PIL import Image


class AdapterError(ValueError):
    """A contract cannot be transformed without losing required meaning."""


@dataclass(frozen=True, slots=True)
class ImageIdentity:
    sha256: str
    width: int
    height: int
    document_id: str
    page_id: str
    page_number: int

    @classmethod
    def from_bytes(
        cls,
        image_bytes: bytes,
        *,
        document_id: str,
        page_id: str,
        page_number: int,
    ) -> "ImageIdentity":
        try:
            with Image.open(io.BytesIO(image_bytes)) as image:
                image.load()
                width, height = image.size
        except OSError as exc:
            raise AdapterError("The canonical page image is unreadable") from exc
        if not re.fullmatch(r"page_[0-9]{3}", page_id):
            raise AdapterError("page_id must use page_### format")
        if page_number < 1:
            raise AdapterError("page_number must be positive")
        return cls(
            sha256=hashlib.sha256(image_bytes).hexdigest(),
            width=width,
            height=height,
            document_id=document_id,
            page_id=page_id,
            page_number=page_number,
        )


def _finite_number(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AdapterError(f"{label} must be numeric") from exc
    if result != result or result in {float("inf"), float("-inf")}:
        raise AdapterError(f"{label} must be finite")
    return result


def validate_xyxy(box: Iterable[Any], width: int, height: int, label: str = "bbox") -> list[float]:
    values = list(box)
    if len(values) != 4:
        raise AdapterError(f"{label} must contain four coordinates")
    x1, y1, x2, y2 = (_finite_number(value, label) for value in values)
    if x2 <= x1 or y2 <= y1:
        raise AdapterError(f"{label} must have positive width and height")
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        raise AdapterError(f"{label} lies outside the {width}x{height} page")
    return [x1, y1, x2, y2]


def normalized_xyxy_to_pixels(box: Iterable[Any], width: int, height: int) -> list[float]:
    values = list(box)
    if len(values) != 4:
        raise AdapterError("normalized bbox must contain four coordinates")
    normalized = [_finite_number(value, "normalized bbox") for value in values]
    if any(value < 0 or value > 1 for value in normalized):
        raise AdapterError("normalized bbox coordinates must be between 0 and 1")
    return validate_xyxy(
        [normalized[0] * width, normalized[1] * height, normalized[2] * width, normalized[3] * height],
        width,
        height,
    )


def xyxy_to_xywh(box: Iterable[Any], width: int, height: int) -> dict[str, float]:
    x1, y1, x2, y2 = validate_xyxy(box, width, height)
    return {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1}


def xywh_to_xyxy(box: dict[str, Any], width: int, height: int) -> list[float]:
    required = {"x", "y", "width", "height"}
    if not required.issubset(box):
        raise AdapterError("xywh bbox is missing a required coordinate")
    x = _finite_number(box["x"], "bbox.x")
    y = _finite_number(box["y"], "bbox.y")
    w = _finite_number(box["width"], "bbox.width")
    h = _finite_number(box["height"], "bbox.height")
    return validate_xyxy([x, y, x + w, y + h], width, height)


def xyxy_to_normalized_xywh(box: Iterable[Any], width: int, height: int) -> dict[str, float]:
    converted = xyxy_to_xywh(box, width, height)
    return {
        "x": converted["x"] / width,
        "y": converted["y"] / height,
        "width": converted["width"] / width,
        "height": converted["height"] / height,
    }


def normalized_xywh_to_xyxy(box: dict[str, Any], width: int, height: int) -> list[float]:
    values = {key: _finite_number(box.get(key), f"bbox.{key}") for key in ("x", "y", "width", "height")}
    if any(value < 0 or value > 1 for value in values.values()):
        raise AdapterError("normalized xywh coordinates must be between 0 and 1")
    return xywh_to_xyxy(
        {
            "x": values["x"] * width,
            "y": values["y"] * height,
            "width": values["width"] * width,
            "height": values["height"] * height,
        },
        width,
        height,
    )


def _safe_id(prefix: str, value: Any, index: int) -> str:
    raw = str(value or f"{index:04d}")
    cleaned = re.sub(r"[^A-Za-z0-9_-]", "_", raw)
    return cleaned if cleaned.startswith(prefix) else f"{prefix}{cleaned}"


def _language(value: Any, text: str) -> tuple[str, str]:
    supplied = str(value or "").lower()
    has_myanmar = bool(re.search(r"[\u1000-\u109f\uaa60-\uaa7f]", text))
    has_latin = bool(re.search(r"[A-Za-z]", text))
    if supplied in {"mya", "my", "burmese"} or (has_myanmar and not has_latin):
        return "my", "Myanmar"
    if supplied in {"eng", "en", "english"} or (has_latin and not has_myanmar):
        return "en", "Latin"
    if supplied == "mixed" or (has_myanmar and has_latin):
        return "mixed", "Mixed"
    return "unknown", "Unknown"


LAYOUT_TYPES = {
    "text_input_box": "TEXT_INPUT_BOX",
    "multiline_box": "MULTILINE_BOX",
    "input_line": "INPUT_LINE",
    "checkbox": "CHECKBOX",
    "radio_button": "RADIO_BUTTON",
    "signature_area": "SIGNATURE_AREA",
    "table": "TABLE",
    "table_cell": "TABLE_CELL",
    "photo_area": "PHOTO_AREA",
    "stamp_area": "STAMP_AREA",
    "character_box_group": "CHARACTER_BOX_GROUP",
}


def _ocr_bbox(token: dict[str, Any], width: int, height: int) -> list[float]:
    if "bbox_px" in token:
        return validate_xyxy(token["bbox_px"], width, height, f"token {token.get('token_id')} bbox")
    box = token.get("bounding_box", token.get("normalized_box"))
    if box is None:
        raise AdapterError(f"token {token.get('token_id')} has no geometry")
    values = list(box)
    # The OCR repository currently emits pixels under `bounding_box`; older examples
    # emitted normalized values. Detect only the unambiguous normalized range.
    if values and all(0 <= _finite_number(value, "token bbox") <= 1 for value in values):
        return normalized_xyxy_to_pixels(values, width, height)
    return validate_xyxy(values, width, height, f"token {token.get('token_id')} bbox")


def build_vlm_contracts(
    ocr_response: dict[str, Any],
    layout_response: dict[str, Any],
    identity: ImageIdentity,
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    """Build strict Insurance-VLM contracts around one authoritative page identity."""
    ocr_pages = ocr_response.get("pages") or []
    layout_pages = layout_response.get("pages") or []
    if len(ocr_pages) != 1 or len(layout_pages) != 1:
        raise AdapterError("Insurance-VLM v1 accepts exactly one OCR and layout page")

    warnings: list[str] = []
    tokens: list[dict[str, Any]] = []
    seen_tokens: set[str] = set()
    for index, source in enumerate(ocr_pages[0].get("tokens", []), 1):
        token_id = _safe_id("token_", source.get("token_id", source.get("id")), index)
        if token_id in seen_tokens:
            raise AdapterError(f"duplicate OCR token ID after normalization: {token_id}")
        seen_tokens.add(token_id)
        text = unicodedata.normalize("NFC", str(source.get("text", "")).strip())
        normalized = unicodedata.normalize("NFC", str(source.get("normalized_text", text)).strip())
        language, script = _language(source.get("language"), text)
        confidence = _finite_number(source.get("confidence", 0), f"{token_id} confidence")
        if confidence > 1:
            confidence /= 100
        if not 0 <= confidence <= 1:
            raise AdapterError(f"{token_id} confidence is outside 0..1")
        tokens.append(
            {
                "token_id": token_id,
                "text": text,
                "normalized_text": normalized,
                "language": language,
                "script": script,
                "bbox_px": _ocr_bbox(source, identity.width, identity.height),
                "confidence": confidence,
                "reading_order": int(source.get("reading_order", index - 1)),
                "line_id": source.get("line_id"),
                "block_id": source.get("block_id"),
            }
        )

    raw_regions = layout_pages[0].get("regions", [])
    id_map = {
        str(region.get("region_id")): _safe_id("region_", region.get("region_id"), index)
        for index, region in enumerate(raw_regions, 1)
    }
    regions: list[dict[str, Any]] = []
    seen_regions: set[str] = set()
    for index, source in enumerate(raw_regions, 1):
        class_name = str(source.get("class_name", source.get("region_type", ""))).lower()
        region_type = LAYOUT_TYPES.get(class_name)
        if region_type is None:
            warnings.append(f"layout region {source.get('region_id')} class {class_name!r} is not actionable and was omitted")
            continue
        region_id = id_map[str(source.get("region_id"))]
        if region_id in seen_regions:
            raise AdapterError(f"duplicate layout region ID after normalization: {region_id}")
        seen_regions.add(region_id)
        parent_raw = source.get("parent_region_id")
        parent_id = id_map.get(str(parent_raw)) if parent_raw is not None else None
        if region_type == "TABLE_CELL":
            if parent_id is None:
                raise AdapterError(f"{region_id}: TABLE_CELL is missing its parent")
            parent = next((item for item in raw_regions if str(item.get("region_id")) == str(parent_raw)), None)
            parent_type = str((parent or {}).get("class_name", (parent or {}).get("region_type", ""))).lower()
            if LAYOUT_TYPES.get(parent_type) != "TABLE":
                raise AdapterError(f"{region_id}: TABLE_CELL parent must be TABLE")
        box = source.get("bbox_px")
        if box is None:
            raise AdapterError(f"{region_id} has no authoritative bbox_px")
        region: dict[str, Any] = {
            "region_id": region_id,
            "region_type": region_type,
            "bbox_px": validate_xyxy(box, identity.width, identity.height, f"{region_id} bbox"),
            "confidence": max(0.0, min(1.0, _finite_number(source.get("confidence", 0), f"{region_id} confidence"))),
            "read_order": source.get("read_order", source.get("reading_order")),
            "parent_region_id": parent_id,
        }
        if source.get("polygon_px") is not None:
            region["polygon_px"] = source["polygon_px"]
        regions.append(region)

    page_identity = {
        "page_id": identity.page_id,
        "page_number": identity.page_number,
        "image_sha256": identity.sha256,
        "width": identity.width,
        "height": identity.height,
    }
    ocr_contract = {
        "schema_version": "1.0.0",
        "document_id": identity.document_id,
        "model": {
            "name": str((ocr_response.get("model") or {}).get("engine", (ocr_response.get("model") or {}).get("name", "ocr-fastapi-service"))),
            "version": str((ocr_response.get("model") or {}).get("version", "1.0.0")),
            "checkpoint": (ocr_response.get("model") or {}).get("checkpoint"),
        },
        "pages": [{**page_identity, "tokens": tokens}],
    }
    layout_model = layout_response.get("model") or {}
    layout_contract = {
        "schema_version": "1.0.0",
        "document_id": identity.document_id,
        "model": {
            "name": str(layout_model.get("name", "PP-DocLayoutV3")),
            "version": str(layout_model.get("version", "unknown")),
            "checkpoint": layout_model.get("checkpoint"),
        },
        "pages": [{**page_identity, "regions": regions}],
    }
    return ocr_contract, layout_contract, warnings


VLM_FIELD_TYPES = {
    "text": "printed_text",
    "multiline_text": "printed_text",
    "integer": "printed_text",
    "decimal": "printed_text",
    "date": "printed_text",
    "time": "printed_text",
    "boolean": "checkbox",
    "single_choice": "checkbox",
    "multiple_choice": "checkbox",
    "signature": "signature",
    "table": "table",
}

EXTRACTION_MODES = {
    "printed": "printed_text",
    "printed_text": "printed_text",
    "handwriting": "handwriting",
    "handwritten": "handwriting",
    "checkbox": "checkbox",
    "table": "table",
    "signature": "signature",
}


def semantic_draft_to_template(
    *,
    template_id: str,
    name: str,
    width: int,
    height: int,
    regions: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    """Convert a human-approved editable draft into document-processing format."""
    fields: list[dict[str, Any]] = []
    review_flags: list[str] = []
    seen: set[str] = set()
    for index, region in enumerate(regions, 1):
        field_id = _safe_id("field_", region.get("field_id", region.get("id")), index)
        if field_id in seen:
            raise AdapterError(f"duplicate template field ID: {field_id}")
        seen.add(field_id)
        bbox = region.get("bbox")
        if not isinstance(bbox, dict):
            raise AdapterError(f"{field_id} is missing editable bbox geometry")
        box_px = normalized_xywh_to_xyxy(bbox, width, height)
        requested_type = str(region.get("extraction_mode", region.get("field_type", ""))).lower()
        mapped = EXTRACTION_MODES.get(requested_type) or VLM_FIELD_TYPES.get(requested_type)
        if mapped is None:
            review_flags.append(f"{field_id}: unsupported field type {requested_type!r}")
            continue
        fields.append(
            {
                "id": field_id,
                "label": str(region.get("label") or region.get("key") or field_id),
                "field_type": mapped,
                "bbox": xyxy_to_xywh(box_px, width, height),
                "required": bool(region.get("required", False)),
                "validation_regex": (region.get("validation") or {}).get("pattern") if isinstance(region.get("validation"), dict) else None,
            }
        )
    if not fields:
        raise AdapterError("The approved draft contains no supported fields with geometry")
    return {
        "template_id": template_id,
        "name": name,
        "width": width,
        "height": height,
        "fields": fields,
    }, review_flags
