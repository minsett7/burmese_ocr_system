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
    validate_vlm_relationships,
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
    page_identity = identity()
    return {
        "schema_version": "1.0.0",
        "document_id": page_identity.document_id,
        "model": {"engine": "tesseract", "version": "1"},
        "pages": [
            {
                "page_id": page_identity.page_id,
                "page_number": page_identity.page_number,
                "image_sha256": page_identity.sha256,
                "width": page_identity.width,
                "height": page_identity.height,
                "tokens": [
                    {
                        "token_id": "tok_0001",
                        "text": "Policy",
                        "normalized_text": "Policy",
                        "language": "eng",
                        "confidence": 0.9,
                        "reading_order": 0,
                        **(
                            {"bounding_box": box}
                            if box is not None
                            else {"normalized_box": [0.1, 0.1, 0.4, 0.2]}
                        ),
                    }
                ]
            }
        ],
    }


def base_layout(regions=None):
    page_identity = identity()
    return {
        "schema_version": "1.1.0",
        "document_id": page_identity.document_id,
        "coordinate_space": "preprocessed_page_pixels",
        "model": {"name": "PP-DocLayoutV3", "version": "1"},
        "pages": [
            {
                "page_id": page_identity.page_id,
                "page_number": page_identity.page_number,
                "image_sha256": page_identity.sha256,
                "width": page_identity.width,
                "height": page_identity.height,
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


def base_vlm_result(layout_contract):
    records = [
        {
            "region_id": region["region_id"],
            "region_type": region["region_type"],
            "bbox_px": region["bbox_px"],
            "parent_region_id": region.get("parent_region_id"),
            "status": "STRUCTURAL_TABLE_CELL" if region["region_type"] == "TABLE_CELL" else "REVIEW_REQUIRED",
            "field_id": None,
            "semantic_key": None,
            "field_type": "table_cell" if region["region_type"] == "TABLE_CELL" else None,
            "confidence": region["confidence"],
            "needs_review": region["region_type"] != "TABLE_CELL",
            "assignment_review_required": False,
        }
        for region in layout_contract["pages"][0]["regions"]
    ]
    actionable = [item for item in records if item["status"] != "STRUCTURAL_TABLE_CELL"]
    unassigned = [item for item in records if item["status"] == "REVIEW_REQUIRED"]
    structural = [item for item in records if item["status"] == "STRUCTURAL_TABLE_CELL"]
    return {
        "status": "COMPLETED",
        "accepted": True,
        "review_required": True,
        "semantic_output": {
            "schema_version": "1.0.0",
            "document_id": layout_contract["document_id"],
            "status": "REVIEW_REQUIRED",
            "pages": [{
                "page_id": layout_contract["pages"][0]["page_id"],
                "page_number": layout_contract["pages"][0]["page_number"],
                "semantic_labels": [],
                "fields": [],
            }],
            "warnings": [],
        },
        "coverage_output": {
            "document_id": layout_contract["document_id"],
            "page_id": layout_contract["pages"][0]["page_id"],
            "input_region_count": len(records),
            "actionable_region_count": len(actionable),
            "assigned_region_count": 0,
            "assigned_review_region_count": 0,
            "unassigned_region_count": len(unassigned),
            "structural_region_count": len(structural),
            "review_region_count": len(unassigned),
            "records": records,
        },
        "quality_summary": {
            "target_region_count": len(actionable),
            "semantic_field_count": 0,
            "assigned_region_count": 0,
            "assigned_review_region_count": 0,
            "unassigned_region_count": len(unassigned),
            "structural_region_count": len(structural),
            "actionable_coverage_ratio": 0.0,
            "semantic_consistency_warning_count": 0,
            "structured_table_count": sum(
                item["region_type"] == "TABLE"
                for item in layout_contract["pages"][0]["regions"]
            ),
            "mapping_complete": not unassigned,
            "quality_status": "MAPPED" if not unassigned else "INCOMPLETE_REVIEW_REQUIRED",
            "automation_ready": False,
        },
        "table_output": {
            "document_id": layout_contract["document_id"],
            "page_id": layout_contract["pages"][0]["page_id"],
            "table_count": 0,
            "tables": [],
        },
        "consistency_warnings": [],
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


@pytest.mark.parametrize(
    ("source", "field", "bad_value"),
    [
        ("ocr", "document_id", "other-document"),
        ("layout", "document_id", "other-document"),
        ("ocr", "page_id", "page_002"),
        ("layout", "page_number", 2),
        ("ocr", "image_sha256", "0" * 64),
        ("layout", "width", 99),
        ("ocr", "height", 199),
    ],
)
def test_upstream_identity_mismatch_fails(source, field, bad_value):
    ocr = base_ocr()
    layout = base_layout()
    target = ocr if source == "ocr" else layout
    if field == "document_id":
        target[field] = bad_value
    else:
        target["pages"][0][field] = bad_value

    with pytest.raises(AdapterError, match="does not match the canonical"):
        build_vlm_contracts(ocr, layout, identity())


def test_layout_coordinate_space_must_be_canonical_pixels():
    layout = base_layout()
    layout["coordinate_space"] = "normalized"
    with pytest.raises(AdapterError, match="coordinate_space"):
        build_vlm_contracts(base_ocr(), layout, identity())


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


def test_duplicate_layout_ids_fail_even_when_region_is_not_actionable():
    regions = [
        {"region_id": "same", "class_name": "field_label", "confidence": 1, "bbox_px": [1, 1, 10, 10]},
        {"region_id": "same", "class_name": "input_line", "confidence": 1, "bbox_px": [1, 20, 10, 30]},
    ]
    with pytest.raises(AdapterError, match="duplicate layout region ID"):
        build_vlm_contracts(base_ocr(), base_layout(regions), identity())


@pytest.mark.parametrize("source", ["ocr", "layout"])
def test_missing_source_ids_are_rejected_not_invented(source):
    ocr = base_ocr()
    layout = base_layout()
    if source == "ocr":
        ocr["pages"][0]["tokens"][0]["token_id"] = ""
    else:
        layout["pages"][0]["regions"][0]["region_id"] = None
    with pytest.raises(AdapterError, match="ID is required"):
        build_vlm_contracts(ocr, layout, identity())


def test_unknown_parent_reference_is_rejected():
    regions = [{
        "region_id": "cell",
        "class_name": "table_cell",
        "confidence": 1,
        "bbox_px": [1, 1, 10, 10],
        "parent_region_id": "missing-table",
    }]
    with pytest.raises(AdapterError, match="parent_region_id does not exist"):
        build_vlm_contracts(base_ocr(), base_layout(regions), identity())


def test_out_of_bounds_polygon_is_rejected():
    layout = base_layout()
    layout["pages"][0]["regions"][0]["polygon_px"] = [[1, 1], [101, 1], [1, 5]]
    with pytest.raises(AdapterError, match="polygon point 2 lies outside"):
        build_vlm_contracts(base_ocr(), layout, identity())


def test_invalid_layout_confidence_is_rejected_not_clamped():
    layout = base_layout()
    layout["pages"][0]["regions"][0]["confidence"] = 1.5
    with pytest.raises(AdapterError, match="confidence is outside"):
        build_vlm_contracts(base_ocr(), layout, identity())


def test_non_actionable_layout_class_is_flagged_not_invented():
    regions = [{"region_id": "label", "class_name": "field_label", "confidence": 1, "bbox_px": [1, 1, 10, 10]}]
    _, layout, warnings = build_vlm_contracts(base_ocr(), base_layout(regions), identity())
    assert layout["pages"][0]["regions"] == []
    assert "not actionable" in warnings[0]


def test_relationship_validation_explicitly_reports_unassigned_regions():
    ocr, layout, _ = build_vlm_contracts(base_ocr(), base_layout(), identity())
    result = base_vlm_result(layout)

    warnings = validate_vlm_relationships(result, ocr, layout)

    assert warnings == [
        "Unassigned layout regions require human review: region_page_001_region_0001"
    ]


def test_relationship_validation_rejects_unknown_region_reference():
    ocr, layout, _ = build_vlm_contracts(base_ocr(), base_layout(), identity())
    result = base_vlm_result(layout)
    result["semantic_output"]["pages"][0].update({
        "semantic_labels": [{
            "label_id": "label_policy",
            "token_ids": ["token_tok_0001"],
            "confidence": 0.9,
        }],
        "fields": [{
            "field_id": "field_policy",
            "key": "policy",
            "label_id": "label_policy",
            "region_ids": ["region_missing"],
            "field_type": "text",
            "relationship": "RIGHT_OF",
            "confidence": 0.9,
        }],
    })

    with pytest.raises(AdapterError, match="unknown layout region"):
        validate_vlm_relationships(result, ocr, layout)


def test_relationship_validation_rejects_changed_coverage_geometry():
    ocr, layout, _ = build_vlm_contracts(base_ocr(), base_layout(), identity())
    result = base_vlm_result(layout)
    result["coverage_output"]["records"][0]["bbox_px"] = [1, 1, 2, 2]

    with pytest.raises(AdapterError, match="changed authoritative geometry"):
        validate_vlm_relationships(result, ocr, layout)


def test_relationship_validation_requires_complete_coverage():
    ocr, layout, _ = build_vlm_contracts(base_ocr(), base_layout(), identity())
    result = base_vlm_result(layout)
    result["coverage_output"]["records"] = []

    with pytest.raises(AdapterError, match="does not account for layout regions"):
        validate_vlm_relationships(result, ocr, layout)


def test_relationship_validation_rejects_inconsistent_quality_counts():
    ocr, layout, _ = build_vlm_contracts(base_ocr(), base_layout(), identity())
    result = base_vlm_result(layout)
    result["quality_summary"]["unassigned_region_count"] = 0

    with pytest.raises(AdapterError, match="quality_summary unassigned_region_count"):
        validate_vlm_relationships(result, ocr, layout)


def test_relationship_validation_preserves_table_cell_geometry_and_parentage():
    regions = [
        {"region_id": "table-a", "class_name": "table", "confidence": 1, "bbox_px": [1, 1, 99, 199]},
        {"region_id": "cell-a", "class_name": "table_cell", "confidence": 1, "bbox_px": [2, 2, 50, 50], "parent_region_id": "table-a"},
    ]
    ocr, layout, _ = build_vlm_contracts(base_ocr(), base_layout(regions), identity())
    result = base_vlm_result(layout)
    result["table_output"] = {
        "document_id": "doc-1",
        "page_id": "page_001",
        "table_count": 1,
        "tables": [{
            "table_region_id": "region_table-a",
            "bbox_px": [1.0, 1.0, 99.0, 199.0],
            "field_id": None,
            "semantic_key": None,
            "row_count": 1,
            "column_count": 1,
            "cell_count": 1,
            "needs_review": True,
            "cells": [{
                "region_id": "region_cell-a",
                "row_index": 1,
                "column_index": 1,
                "bbox_px": [2.0, 2.0, 50.0, 50.0],
                "token_ids": [],
                "text": "",
                "confidence": 1.0,
                "is_header": True,
            }],
        }],
    }

    warnings = validate_vlm_relationships(result, ocr, layout)
    assert "region_table-a" in warnings[0]

    result["table_output"]["tables"][0]["cells"][0]["bbox_px"] = [3, 3, 50, 50]
    with pytest.raises(AdapterError, match="changed authoritative geometry for region_cell-a"):
        validate_vlm_relationships(result, ocr, layout)


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
    assert all(
        isinstance(value, int)
        for field in definition["fields"]
        for value in field["bbox"].values()
    )


def test_semantic_template_mapping_encloses_fractional_pixel_geometry_with_integers():
    definition, flags = semantic_draft_to_template(
        template_id="template_v1",
        name="Test",
        width=1600,
        height=2100,
        regions=[{
            "id": "fractional-checkbox",
            "key": "vehicle_type_car",
            "extraction_mode": "checkbox",
            "bbox": {
                "x": 0.2459821601296126,
                "y": 0.5840816376663972,
                "width": 0.029940453160516575,
                "height": 0.02589568325441719,
            },
        }],
    )

    assert flags == []
    assert definition["fields"][0]["bbox"] == {
        "x": 393,
        "y": 1226,
        "width": 49,
        "height": 55,
    }


def test_semantic_template_mapping_uses_each_pages_dimensions():
    definition, flags = semantic_draft_to_template(
        template_id="template_v1",
        name="Two pages",
        width=100,
        height=200,
        page_dimensions=[
            {"page_number": 1, "width": 100, "height": 200},
            {"page_number": 2, "width": 300, "height": 400},
        ],
        regions=[
            {
                "id": "page-one",
                "page": 1,
                "key": "page_one",
                "extraction_mode": "printed_text",
                "bbox": {"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2},
            },
            {
                "id": "page-two",
                "page": 2,
                "key": "page_two",
                "extraction_mode": "printed_text",
                "bbox": {"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2},
            },
        ],
    )

    assert flags == []
    assert definition["pages"] == [
        {"page_number": 1, "width": 100, "height": 200},
        {"page_number": 2, "width": 300, "height": 400},
    ]
    assert definition["fields"][0]["page"] == 1
    assert definition["fields"][0]["bbox"] == {
        "x": 10, "y": 20, "width": 20, "height": 40,
    }
    assert definition["fields"][1]["page"] == 2
    assert definition["fields"][1]["bbox"] == {
        "x": 30, "y": 40, "width": 60, "height": 80,
    }


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


def test_disabled_draft_region_is_omitted_from_approved_template():
    regions = [
        {
            "id": "enabled",
            "key": "enabled",
            "extraction_mode": "printed_text",
            "bbox": {"x": 0, "y": 0, "width": 0.1, "height": 0.1},
        },
        {
            "id": "disabled",
            "key": "disabled",
            "extraction_mode": None,
            "bbox": {"x": 0.2, "y": 0, "width": 0.1, "height": 0.1},
            "enabled": False,
        },
    ]
    definition, flags = semantic_draft_to_template(
        template_id="template_v1", name="Test", width=100, height=200, regions=regions
    )
    assert flags == []
    assert [field["id"] for field in definition["fields"]] == ["field_enabled"]
