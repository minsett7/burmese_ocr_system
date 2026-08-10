from __future__ import annotations

import hashlib
import io

import pytest
from PIL import Image

from adapters.contracts import (
    AdapterError,
    ImageIdentity,
    build_vlm_contracts,
    normalized_xyxy_to_pixels,
    semantic_draft_to_template,
    xywh_to_xyxy,
    xyxy_to_normalized_xywh,
    xyxy_to_xywh,
)


def image_bytes(width: int = 100, height: int = 200) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(output, format="PNG")
    return output.getvalue()


def identity() -> ImageIdentity:
    return ImageIdentity.from_bytes(
        image_bytes(), document_id="doc-1", page_id="page_001", page_number=1
    )


def base_ocr(box=None):
    return {
        "model": {"engine": "tesseract", "version": "1"},
        "pages": [
            {
                "tokens": [
                    {
                        "token_id": "tok_0001",
                        "text": "Policy",
                        "normalized_text": "Policy",
                        "language": "eng",
                        "confidence": 0.9,
                        "reading_order": 0,
                        "bounding_box": box or [0.1, 0.1, 0.4, 0.2],
                    }
                ]
            }
        ],
    }


def base_layout(regions=None):
    return {
        "model": {"name": "PP-DocLayoutV3", "version": "1"},
        "pages": [
            {
                "regions": regions
                or [
                    {
                        "region_id": "page_001_region_0001",
                        "class_name": "input_line",
                        "confidence": 0.8,
                        "bbox_px": [10, 40, 90, 60],
                        "reading_order": 0,
                        "parent_region_id": None,
                    }
                ]
            }
        ],
    }


def test_bbox_conversions_round_trip():
    box = [10, 20, 70, 80]
    assert xyxy_to_xywh(box, 100, 200) == {"x": 10, "y": 20, "width": 60, "height": 60}
    assert xywh_to_xyxy({"x": 10, "y": 20, "width": 60, "height": 60}, 100, 200) == box
    normalized = xyxy_to_normalized_xywh(box, 100, 200)
    assert normalized == {"x": 0.1, "y": 0.1, "width": 0.6, "height": 0.3}
    assert normalized_xyxy_to_pixels([0.1, 0.1, 0.7, 0.4], 100, 200) == box


def test_ocr_normalized_boxes_are_converted_using_exact_dimensions():
    ocr, _, _ = build_vlm_contracts(base_ocr(), base_layout(), identity())
    assert ocr["pages"][0]["tokens"][0]["bbox_px"] == [10, 20, 40, 40]


def test_pixel_bounding_box_is_not_scaled_twice():
    ocr, _, _ = build_vlm_contracts(base_ocr([10, 20, 40, 40]), base_layout(), identity())
    assert ocr["pages"][0]["tokens"][0]["bbox_px"] == [10, 20, 40, 40]


def test_ocr_and_layout_share_authoritative_identity():
    page = image_bytes()
    expected = hashlib.sha256(page).hexdigest()
    ocr, layout, _ = build_vlm_contracts(base_ocr(), base_layout(), identity())
    for contract in (ocr, layout):
        assert contract["document_id"] == "doc-1"
        page_record = contract["pages"][0]
        assert page_record["page_id"] == "page_001"
        assert page_record["page_number"] == 1
        assert page_record["image_sha256"] == expected
        assert page_record["width"] == 100
        assert page_record["height"] == 200


def test_stable_ids_and_table_parent_are_preserved_deterministically():
    regions = [
        {"region_id": "table-a", "class_name": "table", "confidence": 1, "bbox_px": [1, 1, 99, 199]},
        {"region_id": "cell-a", "class_name": "table_cell", "confidence": 1, "bbox_px": [2, 2, 50, 50], "parent_region_id": "table-a"},
    ]
    ocr, layout, _ = build_vlm_contracts(base_ocr(), base_layout(regions), identity())
    assert ocr["pages"][0]["tokens"][0]["token_id"] == "token_tok_0001"
    assert [item["region_id"] for item in layout["pages"][0]["regions"]] == ["region_table-a", "region_cell-a"]
    assert layout["pages"][0]["regions"][1]["parent_region_id"] == "region_table-a"


@pytest.mark.parametrize("box", ([0, 0, 101, 20], [20, 20, 10, 30]))
def test_out_of_bounds_and_negative_area_boxes_fail(box):
    with pytest.raises(AdapterError):
        build_vlm_contracts(base_ocr(box), base_layout(), identity())


def test_missing_table_parent_fails():
    regions = [{"region_id": "cell", "class_name": "table_cell", "confidence": 1, "bbox_px": [1, 1, 10, 10]}]
    with pytest.raises(AdapterError, match="missing its parent"):
        build_vlm_contracts(base_ocr(), base_layout(regions), identity())


def test_duplicate_ids_fail():
    token = base_ocr()["pages"][0]["tokens"][0]
    payload = base_ocr()
    payload["pages"][0]["tokens"] = [token, dict(token)]
    with pytest.raises(AdapterError, match="duplicate OCR token"):
        build_vlm_contracts(payload, base_layout(), identity())


def test_non_actionable_layout_class_is_flagged_not_invented():
    regions = [{"region_id": "label", "class_name": "field_label", "confidence": 1, "bbox_px": [1, 1, 10, 10]}]
    _, layout, warnings = build_vlm_contracts(base_ocr(), base_layout(regions), identity())
    assert layout["pages"][0]["regions"] == []
    assert "not actionable" in warnings[0]


def test_semantic_template_mapping_supports_all_document_field_types():
    modes = ["printed_text", "handwriting", "checkbox", "table", "signature"]
    regions = [
        {
            "id": f"r-{index}",
            "key": f"field_{index}",
            "label": mode,
            "extraction_mode": mode,
            "bbox": {"x": 0.1, "y": index * 0.1, "width": 0.2, "height": 0.05},
        }
        for index, mode in enumerate(modes, 1)
    ]
    definition, flags = semantic_draft_to_template(
        template_id="template_v1", name="Test", width=100, height=200, regions=regions
    )
    assert flags == []
    assert [field["field_type"] for field in definition["fields"]] == modes


def test_unsupported_field_type_is_flagged_for_human_review():
    regions = [
        {"id": "ok", "key": "ok", "extraction_mode": "printed", "bbox": {"x": 0, "y": 0, "width": 0.1, "height": 0.1}},
        {"id": "photo", "key": "photo", "extraction_mode": "photo", "bbox": {"x": 0.2, "y": 0, "width": 0.1, "height": 0.1}},
    ]
    definition, flags = semantic_draft_to_template(
        template_id="template_v1", name="Test", width=100, height=200, regions=regions
    )
    assert len(definition["fields"]) == 1
    assert "unsupported field type" in flags[0]
