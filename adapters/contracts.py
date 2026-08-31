from __future__ import annotations

import hashlib
import io
import math
import re
import unicodedata
from copy import deepcopy
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


def xyxy_to_integer_xywh(box: Iterable[Any], width: int, height: int) -> dict[str, int]:
    """Enclose a valid pixel box using the integer contract required downstream."""
    x1, y1, x2, y2 = validate_xyxy(box, width, height)
    left = max(0, math.floor(x1))
    top = max(0, math.floor(y1))
    right = min(width, math.ceil(x2))
    bottom = min(height, math.ceil(y2))
    if right <= left or bottom <= top:
        raise AdapterError("integer bbox must have positive width and height")
    return {
        "x": left,
        "y": top,
        "width": right - left,
        "height": bottom - top,
    }


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


def _required_source_id(value: Any, label: str) -> str:
    if value is None or not str(value).strip():
        raise AdapterError(f"{label} is required")
    return str(value).strip()


def _nonnegative_integer(value: Any, label: str, *, default: int | None = None) -> int:
    if value is None and default is not None:
        return default
    if isinstance(value, bool):
        raise AdapterError(f"{label} must be a non-negative integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise AdapterError(f"{label} must be a non-negative integer") from exc
    if result < 0 or result != value:
        raise AdapterError(f"{label} must be a non-negative integer")
    return result


def _confidence(value: Any, label: str, *, accept_percent: bool = False) -> float:
    result = _finite_number(value, label)
    if accept_percent and 1 < result <= 100:
        result /= 100
    if not 0 <= result <= 1:
        raise AdapterError(f"{label} is outside 0..1")
    return result


def _validate_source_page(
    response: dict[str, Any],
    page: dict[str, Any],
    identity: ImageIdentity,
    source_name: str,
) -> None:
    if response.get("document_id") != identity.document_id:
        raise AdapterError(f"{source_name} document_id does not match the canonical document")
    expected = {
        "page_id": identity.page_id,
        "page_number": identity.page_number,
        "image_sha256": identity.sha256,
        "width": identity.width,
        "height": identity.height,
    }
    for field, expected_value in expected.items():
        actual = page.get(field)
        if field == "image_sha256" and isinstance(actual, str):
            actual = actual.lower()
        if actual != expected_value:
            raise AdapterError(
                f"{source_name} {field} does not match the canonical page"
            )


def _validate_polygon(
    polygon: Any, width: int, height: int, label: str
) -> list[list[float]]:
    if not isinstance(polygon, list) or len(polygon) < 3:
        raise AdapterError(f"{label} must contain at least three points")
    result: list[list[float]] = []
    for index, point in enumerate(polygon, 1):
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise AdapterError(f"{label} point {index} must contain x and y")
        x = _finite_number(point[0], f"{label} point {index}")
        y = _finite_number(point[1], f"{label} point {index}")
        if not 0 <= x <= width or not 0 <= y <= height:
            raise AdapterError(f"{label} point {index} lies outside the page")
        result.append([x, y])
    return result


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
    if "bounding_box" in token:
        return validate_xyxy(
            token["bounding_box"], width, height, f"token {token.get('token_id')} bbox"
        )
    if "normalized_box" in token:
        return normalized_xyxy_to_pixels(token["normalized_box"], width, height)
    raise AdapterError(f"token {token.get('token_id')} has no geometry")


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

    ocr_page = ocr_pages[0]
    layout_page = layout_pages[0]
    if not isinstance(ocr_page, dict) or not isinstance(layout_page, dict):
        raise AdapterError("OCR and layout pages must be objects")
    _validate_source_page(ocr_response, ocr_page, identity, "OCR")
    _validate_source_page(layout_response, layout_page, identity, "layout")
    coordinate_space = layout_response.get("coordinate_space")
    if coordinate_space != "preprocessed_page_pixels":
        raise AdapterError("layout coordinate_space must be preprocessed_page_pixels")

    warnings: list[str] = []
    tokens: list[dict[str, Any]] = []
    seen_tokens: set[str] = set()
    for index, source in enumerate(ocr_page.get("tokens", []), 1):
        if not isinstance(source, dict):
            raise AdapterError(f"OCR token {index} must be an object")
        raw_token_id = _required_source_id(
            source.get("token_id", source.get("id")), f"OCR token {index} ID"
        )
        token_id = _safe_id("token_", raw_token_id, index)
        if token_id in seen_tokens:
            raise AdapterError(f"duplicate OCR token ID after normalization: {token_id}")
        seen_tokens.add(token_id)
        text = unicodedata.normalize("NFC", str(source.get("text", "")).strip())
        normalized = unicodedata.normalize("NFC", str(source.get("normalized_text", text)).strip())
        language, script = _language(source.get("language"), text)
        confidence = _confidence(
            source.get("confidence", 0), f"{token_id} confidence", accept_percent=True
        )
        token = {
                "token_id": token_id,
                "text": text,
                "normalized_text": normalized,
                "language": language,
                "script": script,
                "bbox_px": _ocr_bbox(source, identity.width, identity.height),
                "confidence": confidence,
                "reading_order": _nonnegative_integer(
                    source.get("reading_order"), f"{token_id} reading_order", default=index - 1
                ),
                "line_id": source.get("line_id"),
                "block_id": source.get("block_id"),
            }
        if source.get("polygon_px") is not None:
            token["polygon_px"] = _validate_polygon(
                source["polygon_px"], identity.width, identity.height, f"{token_id} polygon"
            )
        tokens.append(token)

    raw_regions = layout_page.get("regions", [])
    if not isinstance(raw_regions, list):
        raise AdapterError("layout regions must be a list")
    id_map: dict[str, str] = {}
    normalized_region_ids: set[str] = set()
    for index, region in enumerate(raw_regions, 1):
        if not isinstance(region, dict):
            raise AdapterError(f"layout region {index} must be an object")
        raw_id = _required_source_id(region.get("region_id"), f"layout region {index} ID")
        normalized_id = _safe_id("region_", raw_id, index)
        if raw_id in id_map or normalized_id in normalized_region_ids:
            raise AdapterError(f"duplicate layout region ID after normalization: {normalized_id}")
        id_map[raw_id] = normalized_id
        normalized_region_ids.add(normalized_id)

    regions: list[dict[str, Any]] = []
    for index, source in enumerate(raw_regions, 1):
        raw_region_id = str(source["region_id"]).strip()
        region_id = id_map[raw_region_id]
        box = source.get("bbox_px")
        if box is None:
            raise AdapterError(f"{region_id} has no authoritative bbox_px")
        validated_box = validate_xyxy(
            box, identity.width, identity.height, f"{region_id} bbox"
        )
        class_name = str(source.get("class_name", source.get("region_type", ""))).lower()
        region_type = LAYOUT_TYPES.get(class_name)
        if region_type is None:
            warnings.append(f"layout region {source.get('region_id')} class {class_name!r} is not actionable and was omitted")
            continue
        parent_raw = source.get("parent_region_id")
        parent_key = str(parent_raw).strip() if parent_raw is not None else None
        parent_id = id_map.get(parent_key) if parent_key else None
        if parent_key and parent_id is None:
            raise AdapterError(f"{region_id}: parent_region_id does not exist")
        if region_type == "TABLE_CELL":
            if parent_id is None:
                raise AdapterError(f"{region_id}: TABLE_CELL is missing its parent")
            parent = next((item for item in raw_regions if str(item.get("region_id")) == str(parent_raw)), None)
            parent_type = str((parent or {}).get("class_name", (parent or {}).get("region_type", ""))).lower()
            if LAYOUT_TYPES.get(parent_type) != "TABLE":
                raise AdapterError(f"{region_id}: TABLE_CELL parent must be TABLE")
        region: dict[str, Any] = {
            "region_id": region_id,
            "region_type": region_type,
            "bbox_px": validated_box,
            "confidence": _confidence(source.get("confidence", 0), f"{region_id} confidence"),
            "read_order": (
                None
                if source.get("read_order", source.get("reading_order")) is None
                else _nonnegative_integer(
                    source.get("read_order", source.get("reading_order")),
                    f"{region_id} read_order",
                )
            ),
            "parent_region_id": parent_id,
        }
        if region_type == "TABLE_CELL":
            for metadata_key in ("row_index", "column_index", "table_cell_order"):
                if source.get(metadata_key) is not None:
                    region[metadata_key] = _nonnegative_integer(
                        source[metadata_key], f"{region_id} {metadata_key}"
                    )
            if source.get("detector") is not None:
                region["detector"] = str(source["detector"])
        if source.get("polygon_px") is not None:
            region["polygon_px"] = _validate_polygon(
                source["polygon_px"], identity.width, identity.height, f"{region_id} polygon"
            )
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

VLM_SEMANTIC_FIELD_TYPES = set(VLM_FIELD_TYPES) | {"photo", "stamp"}

FIELD_REGION_TYPES = {
    "text": {"TEXT_INPUT_BOX", "INPUT_LINE", "CHARACTER_BOX_GROUP"},
    "multiline_text": {"MULTILINE_BOX", "TEXT_INPUT_BOX", "INPUT_LINE", "CHARACTER_BOX_GROUP"},
    "integer": {"TEXT_INPUT_BOX", "INPUT_LINE", "CHARACTER_BOX_GROUP"},
    "decimal": {"TEXT_INPUT_BOX", "INPUT_LINE", "CHARACTER_BOX_GROUP"},
    "date": {"TEXT_INPUT_BOX", "INPUT_LINE", "CHARACTER_BOX_GROUP"},
    "time": {"TEXT_INPUT_BOX", "INPUT_LINE", "CHARACTER_BOX_GROUP"},
    "boolean": {"CHECKBOX"},
    "single_choice": {"RADIO_BUTTON"},
    "multiple_choice": {"CHECKBOX"},
    "signature": {"SIGNATURE_AREA"},
    "table": {"TABLE", "TABLE_CELL"},
    "photo": {"PHOTO_AREA"},
    "stamp": {"STAMP_AREA"},
}

VLM_RELATIONSHIPS = {
    "BELOW", "RIGHT_OF", "INSIDE", "GROUP_BELOW", "CHECKBOX_BEFORE",
    "CHECKBOX_AFTER", "RADIO_BEFORE", "RADIO_AFTER", "TABLE_BELOW",
    "SIGNATURE_BELOW",
}


def _unique_required_strings(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise AdapterError(f"{label} must be a non-empty list")
    result = [_required_source_id(item, label) for item in value]
    if len(result) != len(set(result)):
        raise AdapterError(f"{label} contains duplicate IDs")
    return result


def validate_vlm_relationships(
    result: dict[str, Any],
    ocr_contract: dict[str, Any],
    layout_contract: dict[str, Any],
) -> list[str]:
    """Validate VLM semantics and coverage without allowing geometry replacement."""
    if (
        result.get("status") != "COMPLETED"
        or result.get("accepted") is not True
        or result.get("review_required") is not True
    ):
        raise AdapterError("Insurance-VLM result is not an accepted completed result")

    document_id = layout_contract["document_id"]
    ocr_page = ocr_contract["pages"][0]
    layout_page = layout_contract["pages"][0]
    tokens = {item["token_id"]: item for item in ocr_page["tokens"]}
    regions = {item["region_id"]: item for item in layout_page["regions"]}

    semantic = result.get("semantic_output")
    if not isinstance(semantic, dict) or semantic.get("document_id") != document_id:
        raise AdapterError("VLM semantic document_id does not match the canonical document")
    if semantic.get("schema_version") != "1.0.0" or semantic.get("status") != "REVIEW_REQUIRED":
        raise AdapterError("VLM semantic output has an unsupported schema or status")
    semantic_pages = semantic.get("pages")
    if not isinstance(semantic_pages, list) or len(semantic_pages) != 1:
        raise AdapterError("VLM semantic output must contain exactly one page")
    semantic_page = semantic_pages[0]
    if (
        not isinstance(semantic_page, dict)
        or semantic_page.get("page_id") != layout_page["page_id"]
        or semantic_page.get("page_number") != layout_page["page_number"]
    ):
        raise AdapterError("VLM semantic page identity does not match the canonical page")

    labels_list = semantic_page.get("semantic_labels")
    fields = semantic_page.get("fields")
    if not isinstance(labels_list, list) or not isinstance(fields, list):
        raise AdapterError("VLM semantic labels and fields must be lists")
    labels: dict[str, dict[str, Any]] = {}
    for index, label in enumerate(labels_list, 1):
        if not isinstance(label, dict):
            raise AdapterError(f"VLM semantic label {index} must be an object")
        label_id = _required_source_id(label.get("label_id"), f"VLM label {index} ID")
        if not re.fullmatch(r"label_[A-Za-z0-9_-]+", label_id):
            raise AdapterError(f"invalid VLM label ID: {label_id}")
        if label_id in labels:
            raise AdapterError(f"duplicate VLM label ID: {label_id}")
        for token_id in _unique_required_strings(label.get("token_ids"), f"{label_id} token_ids"):
            if token_id not in tokens:
                raise AdapterError(f"{label_id} references unknown OCR token {token_id}")
        _confidence(label.get("confidence"), f"{label_id} confidence")
        labels[label_id] = label

    field_ids: set[str] = set()
    keys: set[str] = set()
    region_assignments: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    strict_region_types = {
        "CHECKBOX", "RADIO_BUTTON", "TABLE", "TABLE_CELL", "SIGNATURE_AREA",
        "PHOTO_AREA", "STAMP_AREA",
    }
    strict_field_types = {
        "boolean", "single_choice", "multiple_choice", "table", "signature", "photo", "stamp",
    }
    for index, field in enumerate(fields, 1):
        if not isinstance(field, dict):
            raise AdapterError(f"VLM field {index} must be an object")
        field_id = _required_source_id(field.get("field_id"), f"VLM field {index} ID")
        key = _required_source_id(field.get("key"), f"{field_id} key")
        if not re.fullmatch(r"field_[A-Za-z0-9_-]+", field_id):
            raise AdapterError(f"invalid VLM field ID: {field_id}")
        if not re.fullmatch(r"[a-z][a-z0-9_]*", key):
            raise AdapterError(f"{field_id} has invalid semantic key {key!r}")
        if field_id in field_ids:
            raise AdapterError(f"duplicate VLM field ID: {field_id}")
        if key in keys:
            raise AdapterError(f"duplicate VLM field key: {key}")
        field_ids.add(field_id)
        keys.add(key)
        if field.get("label_id") not in labels:
            raise AdapterError(f"{field_id} references unknown semantic label")
        field_type = str(field.get("field_type") or "")
        if field_type not in VLM_SEMANTIC_FIELD_TYPES:
            raise AdapterError(f"{field_id} has unsupported VLM field type {field_type!r}")
        if field.get("relationship") not in VLM_RELATIONSHIPS:
            raise AdapterError(f"{field_id} has an invalid label/field relationship")
        _confidence(field.get("confidence"), f"{field_id} confidence")
        region_ids = _unique_required_strings(field.get("region_ids"), f"{field_id} region_ids")
        for region_id in region_ids:
            region = regions.get(region_id)
            if region is None:
                raise AdapterError(f"{field_id} references unknown layout region {region_id}")
            if region_id in region_assignments:
                raise AdapterError(f"layout region {region_id} is assigned to multiple VLM fields")
            region_assignments[region_id] = field
            if region["region_type"] not in FIELD_REGION_TYPES[field_type]:
                message = (
                    f"{field_id} type {field_type} is incompatible with "
                    f"region {region_id} type {region['region_type']}"
                )
                if region["region_type"] in strict_region_types or field_type in strict_field_types:
                    raise AdapterError(message)
                warnings.append(message)
        for region_id in region_ids:
            region = regions[region_id]
            if (
                region["region_type"] == "TABLE_CELL"
                and str(region.get("parent_region_id") or "") not in region_ids
            ):
                raise AdapterError(
                    f"{field_id} cannot map TABLE_CELL {region_id} without its parent TABLE"
                )
        option_keys: set[str] = set()
        option_controls: set[str] = set()
        for option_index, option in enumerate(field.get("options") or [], 1):
            if not isinstance(option, dict):
                raise AdapterError(f"{field_id} option {option_index} must be an object")
            option_key = _required_source_id(option.get("option_key"), f"{field_id} option key")
            if option_key in option_keys:
                raise AdapterError(f"{field_id} has duplicate option key {option_key}")
            option_keys.add(option_key)
            control_id = _required_source_id(
                option.get("control_region_id"), f"{field_id} option control_region_id"
            )
            if control_id in option_controls:
                raise AdapterError(f"{field_id} reuses option control {control_id}")
            option_controls.add(control_id)
            if control_id not in region_ids:
                raise AdapterError(f"{field_id} option control {control_id} is not owned by the field")
            if regions[control_id]["region_type"] not in {"CHECKBOX", "RADIO_BUTTON"}:
                raise AdapterError(f"{field_id} option control {control_id} is not a checkbox or radio")
            for token_id in _unique_required_strings(
                option.get("label_token_ids"), f"{field_id} option label_token_ids"
            ):
                if token_id not in tokens:
                    raise AdapterError(f"{field_id} option references unknown OCR token {token_id}")
        if field_type not in VLM_FIELD_TYPES:
            warnings.append(f"{field_id} type {field_type} requires human resolution before approval")

    coverage = result.get("coverage_output")
    if not isinstance(coverage, dict):
        raise AdapterError("Insurance-VLM result is missing coverage_output")
    if coverage.get("document_id") != document_id or coverage.get("page_id") != layout_page["page_id"]:
        raise AdapterError("VLM coverage identity does not match the canonical page")
    records = coverage.get("records")
    if not isinstance(records, list):
        raise AdapterError("VLM coverage records must be a list")
    records_by_region: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(records, 1):
        if not isinstance(record, dict):
            raise AdapterError(f"VLM coverage record {index} must be an object")
        region_id = _required_source_id(record.get("region_id"), f"coverage record {index} region_id")
        if region_id in records_by_region:
            raise AdapterError(f"duplicate VLM coverage region ID: {region_id}")
        region = regions.get(region_id)
        if region is None:
            raise AdapterError(f"coverage references unknown layout region {region_id}")
        if record.get("region_type") != region["region_type"]:
            raise AdapterError(f"coverage changed authoritative region type for {region_id}")
        if record.get("bbox_px") != region["bbox_px"]:
            raise AdapterError(f"coverage changed authoritative geometry for {region_id}")
        if record.get("parent_region_id") != region.get("parent_region_id"):
            raise AdapterError(f"coverage changed authoritative parent for {region_id}")
        structural = region["region_type"] == "TABLE_CELL"
        expected_field = (
            region_assignments.get(str(region.get("parent_region_id") or ""))
            if structural
            else region_assignments.get(region_id)
        )
        expected_status = "STRUCTURAL_TABLE_CELL" if structural else "ASSIGNED" if expected_field else "REVIEW_REQUIRED"
        if record.get("status") != expected_status:
            raise AdapterError(f"coverage status is inconsistent for {region_id}")
        if not isinstance(record.get("needs_review"), bool):
            raise AdapterError(f"coverage needs_review must be boolean for {region_id}")
        if expected_status == "REVIEW_REQUIRED" and record["needs_review"] is not True:
            raise AdapterError(f"unassigned coverage region {region_id} must require review")
        if expected_status == "STRUCTURAL_TABLE_CELL" and record["needs_review"] is not False:
            raise AdapterError(f"structural table cell {region_id} cannot be independently reviewable")
        expected_field_id = expected_field.get("field_id") if expected_field else None
        expected_key = expected_field.get("key") if expected_field else None
        if record.get("field_id") != expected_field_id or record.get("semantic_key") != expected_key:
            raise AdapterError(f"coverage field assignment is inconsistent for {region_id}")
        expected_field_type = (
            "table_cell" if structural else expected_field.get("field_type") if expected_field else None
        )
        if record.get("field_type") != expected_field_type:
            raise AdapterError(f"coverage field type is inconsistent for {region_id}")
        records_by_region[region_id] = record

    if set(records_by_region) != set(regions):
        missing = sorted(set(regions) - set(records_by_region))
        raise AdapterError(f"coverage does not account for layout regions: {missing}")
    actionable_records = [
        item for item in records if item.get("status") != "STRUCTURAL_TABLE_CELL"
    ]
    expected_counts = {
        "input_region_count": len(records),
        "actionable_region_count": len(actionable_records),
        "assigned_region_count": sum(item.get("status") == "ASSIGNED" for item in records),
        "assigned_review_region_count": sum(
            item.get("status") == "ASSIGNED" and item.get("needs_review") is True
            for item in records
        ),
        "unassigned_region_count": sum(item.get("status") == "REVIEW_REQUIRED" for item in records),
        "structural_region_count": sum(item.get("status") == "STRUCTURAL_TABLE_CELL" for item in records),
        "review_region_count": sum(item.get("needs_review") is True for item in records),
    }
    for field, expected in expected_counts.items():
        if coverage.get(field) != expected:
            raise AdapterError(f"coverage {field} is inconsistent with its records")

    quality = result.get("quality_summary")
    if not isinstance(quality, dict):
        raise AdapterError("Insurance-VLM result is missing quality_summary")
    quality_counts = {
        "target_region_count": expected_counts["actionable_region_count"],
        "assigned_region_count": expected_counts["assigned_region_count"],
        "unassigned_region_count": expected_counts["unassigned_region_count"],
        "structural_region_count": expected_counts["structural_region_count"],
        "semantic_field_count": len(fields),
        "assigned_review_region_count": expected_counts["assigned_review_region_count"],
    }
    for field, expected in quality_counts.items():
        if quality.get(field) != expected:
            raise AdapterError(f"quality_summary {field} is inconsistent")
    expected_ratio = round(
        expected_counts["assigned_region_count"] / max(1, expected_counts["actionable_region_count"]),
        6,
    )
    if quality.get("actionable_coverage_ratio") != expected_ratio:
        raise AdapterError("quality_summary actionable_coverage_ratio is inconsistent")

    consistency_warnings = [str(item) for item in result.get("consistency_warnings") or []]
    expected_complete = expected_counts["unassigned_region_count"] == 0 and not consistency_warnings
    expected_quality_status = (
        "INCOMPLETE_REVIEW_REQUIRED"
        if not expected_complete
        else "MAPPED_REVIEW_REQUIRED"
        if expected_counts["assigned_review_region_count"]
        else "MAPPED"
    )
    if quality.get("mapping_complete") != expected_complete:
        raise AdapterError("quality_summary mapping_complete is inconsistent")
    if quality.get("quality_status") != expected_quality_status:
        raise AdapterError("quality_summary quality_status is inconsistent")
    if quality.get("semantic_consistency_warning_count") != len(set(consistency_warnings)):
        raise AdapterError("quality_summary semantic_consistency_warning_count is inconsistent")
    if quality.get("automation_ready") is not False:
        raise AdapterError("Insurance-VLM results must remain human-review only")

    table_output = result.get("table_output")
    if not isinstance(table_output, dict):
        raise AdapterError("Insurance-VLM result is missing table_output")
    if table_output.get("document_id") != document_id or table_output.get("page_id") != layout_page["page_id"]:
        raise AdapterError("VLM table output identity does not match the canonical page")
    table_regions = {
        region_id: region
        for region_id, region in regions.items()
        if region["region_type"] == "TABLE"
    }
    cells_by_parent: dict[str, dict[str, dict[str, Any]]] = {
        table_id: {} for table_id in table_regions
    }
    for region_id, region in regions.items():
        if region["region_type"] == "TABLE_CELL":
            cells_by_parent[str(region["parent_region_id"])][region_id] = region
    output_tables = table_output.get("tables")
    if not isinstance(output_tables, list) or table_output.get("table_count") != len(output_tables):
        raise AdapterError("VLM table output count is inconsistent")
    output_table_ids: set[str] = set()
    for index, table in enumerate(output_tables, 1):
        if not isinstance(table, dict):
            raise AdapterError(f"VLM table output {index} must be an object")
        table_id = _required_source_id(table.get("table_region_id"), f"table output {index} region_id")
        if table_id in output_table_ids:
            raise AdapterError(f"duplicate VLM table output for {table_id}")
        output_table_ids.add(table_id)
        authoritative_table = table_regions.get(table_id)
        if authoritative_table is None:
            raise AdapterError(f"table output references unknown TABLE region {table_id}")
        if table.get("bbox_px") != authoritative_table["bbox_px"]:
            raise AdapterError(f"table output changed authoritative geometry for {table_id}")
        expected_field = region_assignments.get(table_id)
        expected_field_id = expected_field.get("field_id") if expected_field else None
        expected_key = expected_field.get("key") if expected_field else None
        if table.get("field_id") != expected_field_id or table.get("semantic_key") != expected_key:
            raise AdapterError(f"table output field assignment is inconsistent for {table_id}")
        cells = table.get("cells")
        if not isinstance(cells, list) or table.get("cell_count") != len(cells):
            raise AdapterError(f"table output cell count is inconsistent for {table_id}")
        output_cell_ids: set[str] = set()
        for cell_index, cell in enumerate(cells, 1):
            if not isinstance(cell, dict):
                raise AdapterError(f"{table_id} cell {cell_index} must be an object")
            cell_id = _required_source_id(cell.get("region_id"), f"{table_id} cell region_id")
            if cell_id in output_cell_ids:
                raise AdapterError(f"duplicate table cell output for {cell_id}")
            output_cell_ids.add(cell_id)
            authoritative_cell = cells_by_parent[table_id].get(cell_id)
            if authoritative_cell is None:
                raise AdapterError(f"{table_id} output references unrelated table cell {cell_id}")
            if cell.get("bbox_px") != authoritative_cell["bbox_px"]:
                raise AdapterError(f"table output changed authoritative geometry for {cell_id}")
            row_index = _nonnegative_integer(cell.get("row_index"), f"{cell_id} row_index")
            column_index = _nonnegative_integer(cell.get("column_index"), f"{cell_id} column_index")
            if row_index < 1 or column_index < 1:
                raise AdapterError(f"{cell_id} row and column indexes must be positive")
            for token_id in cell.get("token_ids") or []:
                if token_id not in tokens:
                    raise AdapterError(f"{cell_id} references unknown OCR token {token_id}")
        if output_cell_ids != set(cells_by_parent[table_id]):
            raise AdapterError(f"table output does not preserve every cell for {table_id}")
        if cells:
            if table.get("row_count") != max(cell["row_index"] for cell in cells):
                raise AdapterError(f"table output row count is inconsistent for {table_id}")
            if table.get("column_count") != max(cell["column_index"] for cell in cells):
                raise AdapterError(f"table output column count is inconsistent for {table_id}")
        elif table.get("row_count") != 0 or table.get("column_count") != 0:
            raise AdapterError(f"empty table output dimensions are inconsistent for {table_id}")
    if output_table_ids != set(table_regions):
        raise AdapterError("table output does not account for every TABLE region")
    if quality.get("structured_table_count") != len(table_regions):
        raise AdapterError("quality_summary structured_table_count is inconsistent")

    for index, warning in enumerate(semantic.get("warnings") or [], 1):
        if not isinstance(warning, dict):
            raise AdapterError(f"semantic warning {index} must be an object")
        if warning.get("page_id") != layout_page["page_id"]:
            raise AdapterError(f"semantic warning {index} references an unknown page")
        for token_id in warning.get("token_ids") or []:
            if token_id not in tokens:
                raise AdapterError(f"semantic warning {index} references unknown token {token_id}")
        for region_id in warning.get("region_ids") or []:
            if region_id not in regions:
                raise AdapterError(f"semantic warning {index} references unknown region {region_id}")

    unassigned = sorted(
        region_id
        for region_id, record in records_by_region.items()
        if record.get("status") == "REVIEW_REQUIRED"
    )
    if unassigned:
        warnings.append(f"Unassigned layout regions require human review: {', '.join(unassigned)}")
    warnings.extend(consistency_warnings)
    return list(dict.fromkeys(warnings))

EXTRACTION_MODES = {
    "printed": "printed_text",
    "printed_text": "printed_text",
    "handwriting": "handwriting",
    "handwritten": "handwriting",
    "checkbox": "checkbox",
    "table": "table",
    "signature": "signature",
}


def _cluster_axis(values: list[float], tolerance: float) -> list[int]:
    clusters: list[list[tuple[int, float]]] = []
    for original_index, value in sorted(enumerate(values), key=lambda item: item[1]):
        if not clusters:
            clusters.append([(original_index, value)])
            continue
        center = sum(item[1] for item in clusters[-1]) / len(clusters[-1])
        if abs(value - center) <= tolerance:
            clusters[-1].append((original_index, value))
        else:
            clusters.append([(original_index, value)])
    assignments = [0] * len(values)
    for cluster_index, cluster in enumerate(clusters):
        for original_index, _ in cluster:
            assignments[original_index] = cluster_index
    return assignments


def repair_template_table_grid(definition: dict[str, Any]) -> dict[str, Any]:
    """Recover collapsed/missing cell indices from immutable template geometry."""
    repaired = deepcopy(definition)
    fields = repaired.get("fields")
    if not isinstance(fields, list):
        return repaired
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for field in fields:
        if not isinstance(field, dict) or not field.get("table_parent_field_id"):
            continue
        groups.setdefault(
            (str(field["table_parent_field_id"]), int(field.get("page") or 1)), []
        ).append(field)

    for members in groups.values():
        positions = [
            (field.get("table_row_index"), field.get("table_column_index"))
            for field in members
        ]
        complete_unique_grid = (
            all(
                isinstance(row, int) and row >= 0
                and isinstance(column, int) and column >= 0
                for row, column in positions
            )
            and len(set(positions)) == len(positions)
        )
        if complete_unique_grid:
            continue
        if any(not isinstance(field.get("bbox"), dict) for field in members):
            continue
        widths = [float(field["bbox"].get("width") or 1) for field in members]
        heights = [float(field["bbox"].get("height") or 1) for field in members]
        median_width = sorted(widths)[len(widths) // 2]
        median_height = sorted(heights)[len(heights) // 2]
        rows = _cluster_axis(
            [float(field["bbox"].get("y") or 0) for field in members],
            max(3.0, median_height * 0.42),
        )
        columns = _cluster_axis(
            [float(field["bbox"].get("x") or 0) for field in members],
            max(3.0, median_width * 0.42),
        )
        ordered = sorted(
            range(len(members)),
            key=lambda index: (
                rows[index],
                columns[index],
                float(members[index]["bbox"].get("y") or 0),
                float(members[index]["bbox"].get("x") or 0),
            ),
        )
        order_by_index = {
            member_index: order for order, member_index in enumerate(ordered)
        }
        for index, field in enumerate(members):
            field["table_row_index"] = rows[index]
            field["table_column_index"] = columns[index]
            field["table_cell_order"] = order_by_index[index]
            parent_label = field.get("table_parent_label")
            if parent_label:
                field["label"] = (
                    f"{parent_label} row {rows[index] + 1}, "
                    f"column {columns[index] + 1}"
                )
    return repaired


def semantic_draft_to_template(
    *,
    template_id: str,
    name: str,
    width: int,
    height: int,
    regions: list[dict[str, Any]],
    structural_regions: list[dict[str, Any]] | None = None,
    page_dimensions: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Convert a human-approved editable draft into document-processing format."""
    dimensions_by_page = {1: (width, height)}
    if page_dimensions:
        dimensions_by_page = {}
        for page in page_dimensions:
            try:
                page_number = int(page["page_number"])
                page_width = int(page["width"])
                page_height = int(page["height"])
            except (KeyError, TypeError, ValueError) as exc:
                raise AdapterError("page dimensions must contain positive integer values") from exc
            if page_number < 1 or page_width < 1 or page_height < 1:
                raise AdapterError("page dimensions must contain positive integer values")
            if page_number in dimensions_by_page:
                raise AdapterError(f"duplicate page dimensions for page {page_number}")
            dimensions_by_page[page_number] = (page_width, page_height)
        if sorted(dimensions_by_page) != list(range(1, len(dimensions_by_page) + 1)):
            raise AdapterError("page dimensions must be sequential starting at page 1")
    multi_page = len(dimensions_by_page) > 1
    fields: list[dict[str, Any]] = []
    review_flags: list[str] = []
    seen: set[str] = set()
    table_cells_by_parent: dict[str, list[dict[str, Any]]] = {}
    for cell in structural_regions or []:
        if str(cell.get("region_type") or "").upper() != "TABLE_CELL":
            continue
        parent_id = cell.get("parent_region_id")
        if parent_id:
            table_cells_by_parent.setdefault(str(parent_id), []).append(cell)
    for cells in table_cells_by_parent.values():
        cells.sort(key=lambda item: (
            int(item.get("table_cell_order") or 0),
            int(item.get("row_index") or 0),
            int(item.get("column_index") or 0),
            str(item.get("id") or ""),
        ))
    for index, region in enumerate(regions, 1):
        if region.get("enabled", True) is False:
            continue
        field_id = _safe_id("field_", region.get("field_id", region.get("id")), index)
        if field_id in seen:
            raise AdapterError(f"duplicate template field ID: {field_id}")
        seen.add(field_id)
        bbox = region.get("bbox")
        if not isinstance(bbox, dict):
            raise AdapterError(f"{field_id} is missing editable bbox geometry")
        try:
            page_number = int(region.get("page", 1))
            page_width, page_height = dimensions_by_page[page_number]
        except (KeyError, TypeError, ValueError) as exc:
            raise AdapterError(f"{field_id} references an unknown template page") from exc
        box_px = normalized_xywh_to_xyxy(bbox, page_width, page_height)
        requested_type = str(region.get("extraction_mode", region.get("field_type", ""))).lower()
        mapped = EXTRACTION_MODES.get(requested_type) or VLM_FIELD_TYPES.get(requested_type)
        if mapped is None:
            review_flags.append(f"{field_id}: unsupported field type {requested_type!r}")
            continue
        table_cells = table_cells_by_parent.get(str(region.get("id")), []) if mapped == "table" else []
        if table_cells:
            parent_label = str(region.get("label") or region.get("key") or field_id)
            for cell_index, cell in enumerate(table_cells, 1):
                cell_field_id = _safe_id(
                    "field_", cell.get("id") or f"{field_id}_cell_{cell_index:03d}", cell_index
                )
                if cell_field_id in seen:
                    raise AdapterError(f"duplicate template field ID: {cell_field_id}")
                cell_bbox = cell.get("bbox")
                if not isinstance(cell_bbox, dict):
                    raise AdapterError(f"{cell_field_id} is missing editable bbox geometry")
                try:
                    cell_page = int(cell.get("page", page_number))
                    cell_page_width, cell_page_height = dimensions_by_page[cell_page]
                except (KeyError, TypeError, ValueError) as exc:
                    raise AdapterError(f"{cell_field_id} references an unknown template page") from exc
                cell_box_px = normalized_xywh_to_xyxy(
                    cell_bbox, cell_page_width, cell_page_height
                )
                row_index = int(cell.get("row_index") or 0)
                column_index = int(cell.get("column_index") or 0)
                cell_field = {
                    "id": cell_field_id,
                    "label": f"{parent_label} row {row_index + 1}, column {column_index + 1}",
                    "field_type": "printed_text",
                    "bbox": xyxy_to_integer_xywh(
                        cell_box_px, cell_page_width, cell_page_height
                    ),
                    "required": False,
                    "validation_regex": None,
                    "table_parent_field_id": field_id,
                    "table_parent_label": parent_label,
                    "table_row_index": row_index,
                    "table_column_index": column_index,
                    "table_cell_order": int(cell.get("table_cell_order") or cell_index - 1),
                    "table_is_header": bool(cell.get("is_header", False)),
                }
                if multi_page:
                    cell_field["page"] = cell_page
                fields.append(cell_field)
                seen.add(cell_field_id)
            continue
        field = {
                "id": field_id,
                "label": str(region.get("label") or region.get("key") or field_id),
                "field_type": mapped,
                "bbox": xyxy_to_integer_xywh(box_px, page_width, page_height),
                "required": bool(region.get("required", False)),
                "validation_regex": (region.get("validation") or {}).get("pattern") if isinstance(region.get("validation"), dict) else None,
            }
        choice_mode = str(region.get("data_type") or "").lower()
        if choice_mode in {"single_choice", "multiple_choice"}:
            field["choice_group_id"] = _safe_id("choice_", region.get("semantic_group_field_id", field_id), index)
            field["choice_option_value"] = str(region.get("option_key") or region.get("key") or field_id)
            field["choice_mode"] = choice_mode
        if multi_page:
            field["page"] = page_number
        fields.append(field)
    if not fields:
        raise AdapterError("The approved draft contains no supported fields with geometry")
    definition = {
        "template_id": template_id,
        "name": name,
        "width": width,
        "height": height,
        "fields": fields,
    }
    if multi_page:
        definition["pages"] = [
            {
                "page_number": page_number,
                "width": dimensions[0],
                "height": dimensions[1],
            }
            for page_number, dimensions in sorted(dimensions_by_page.items())
        ]
    return repair_template_table_grid(definition), review_flags
