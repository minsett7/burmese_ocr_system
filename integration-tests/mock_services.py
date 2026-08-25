from __future__ import annotations

import hashlib
import io
from typing import Any

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import Response
from PIL import Image


app = FastAPI(title="Unified integration downstream mocks")


def page_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (100, 200), "white").save(output, format="PNG")
    return output.getvalue()


PAGE = page_bytes()


@app.get("/health")
@app.get("/health/ready")
def health():
    return {"status": "ok", "engine": "mock"}


@app.post("/v1/documents", status_code=201)
async def layout_upload(file: UploadFile = File(...)):
    await file.read()
    return {"document_id": "visual-mock-document", "status": "uploaded"}


@app.post("/v1/documents/{document_id}/preprocess")
def preprocess(document_id: str, payload: dict[str, Any]):
    capture_profile = payload.get("capture_profile", "template")
    return {
        "document_id": document_id,
        "status": "preprocessed",
        "artifact": f"preprocessed/{capture_profile}_manifest.json",
        "summary": {
            "page_count": 1,
            "quality_pass": True,
            "capture_profile": capture_profile,
            "decision": "canonicalize_only",
            "capture_ready": True,
            "retake_required": False,
            "reasons": [],
            "advisories": [],
            "instructions": [],
            "operations": ["rgb_conversion"],
        },
    }


@app.post("/v1/documents/{document_id}/extract")
def extract(document_id: str, payload: dict[str, Any]):
    return {"document_id": document_id, "status": "extracted", "summary": {"region_count": 2}}


@app.get("/v1/documents/{document_id}/result")
def layout_result(document_id: str):
    return {
        "schema_version": "1.1.0",
        "document_id": document_id,
        "coordinate_space": "preprocessed_page_pixels",
        "model": {"name": "PP-DocLayoutV3-mock", "version": "1.0"},
        "pages": [
            {
                "page_id": "page_001", "page_number": 1, "width": 100, "height": 200,
                "image_sha256": hashlib.sha256(PAGE).hexdigest(),
                "image_path": "preprocessed/pages/page_001.png",
                "regions": [
                    {"region_id": "policy-line", "class_name": "input_line", "confidence": 0.95, "bbox_px": [10, 30, 90, 50]},
                    {"region_id": "accept-check", "class_name": "checkbox", "confidence": 0.95, "bbox_px": [10, 70, 30, 90]},
                ],
            }
        ],
    }


@app.get("/v1/documents/{document_id}/artifacts/{artifact_path:path}")
def artifact(document_id: str, artifact_path: str):
    if artifact_path.endswith("_manifest.json"):
        capture_profile = "document" if artifact_path.endswith("document_manifest.json") else "template"
        return {
            "schema_version": "1.3.0",
            "capture_profile": capture_profile,
            "decision": "canonicalize_only",
            "capture_ready": True,
            "retake_required": False,
            "reasons": [],
            "advisories": [],
            "instructions": [],
            "operations": ["rgb_conversion"],
            "page_count": 1,
            "quality_pass": True,
            "pages": [{
                "image_path": "preprocessed/pages/page_001.png",
                "page_id": "page_001",
                "page_number": 1,
                "width": 100,
                "height": 200,
                "sha256": hashlib.sha256(PAGE).hexdigest(),
                "source_to_page_transform": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "operations": ["rgb_conversion"],
                "quality": {"quality_pass": True, "warnings": []},
            }],
        }
    return Response(PAGE, media_type="image/png")


@app.post("/v1/ocr/process")
async def ocr(file: UploadFile = File(...)):
    content = await file.read()
    return {
        "schema_version": "1.0.0", "document_id": "visual-mock-document", "model": {"engine": "mock-ocr", "version": "1"},
        "pages": [{"page_id": "page_001", "page_number": 1, "image_sha256": hashlib.sha256(content).hexdigest(), "width": 100, "height": 200,
                   "tokens": [{"token_id": "tok_0001", "text": "Policy", "normalized_text": "Policy", "language": "eng", "confidence": 0.99, "reading_order": 0, "bounding_box": [10, 10, 40, 20]}]}],
    }


@app.post("/api/v1/registrations", status_code=202)
async def vlm_registration(image: UploadFile = File(...), ocr_json: UploadFile = File(...), layout_json: UploadFile = File(...)):
    await image.read(); await ocr_json.read(); await layout_json.read()
    return {"job_id": "job_mock_vlm", "status": "PENDING"}


@app.get("/api/v1/registrations/job_mock_vlm")
def vlm_status():
    return {"job_id": "job_mock_vlm", "status": "COMPLETED"}


@app.get("/api/v1/registrations/job_mock_vlm/result")
def vlm_result():
    return {
        "job_id": "job_mock_vlm", "status": "COMPLETED", "accepted": True, "review_required": True,
        "semantic_output": {
            "schema_version": "1.0.0", "document_id": "visual-mock-document", "status": "REVIEW_REQUIRED",
            "document_class": None, "template_name": None,
            "pages": [{
                "page_id": "page_001", "page_number": 1,
                "semantic_labels": [{
                    "label_id": "label_policy", "token_ids": ["token_tok_0001"],
                    "semantic_class": "FIELD_LABEL", "primary_text": "Policy", "primary_language": "en",
                    "translations": {"my": None, "en": "Policy"}, "confidence": 0.95,
                }],
                "fields": [
                    {"field_id": "field_policy", "key": "policy", "label_id": "label_policy",
                     "region_ids": ["region_policy-line"], "field_type": "text", "relationship": "RIGHT_OF",
                     "required": False, "confidence": 0.92, "review_notes": []},
                    {"field_id": "field_confirmed", "key": "confirmed", "label_id": "label_policy",
                     "region_ids": ["region_accept-check"], "field_type": "boolean", "relationship": "CHECKBOX_BEFORE",
                     "required": False, "confidence": 0.88, "review_notes": []},
                ],
            }],
            "warnings": [],
        },
        "coverage_output": {
            "schema_version": "1.0.0", "document_id": "visual-mock-document", "page_id": "page_001",
            "input_region_count": 2, "actionable_region_count": 2, "assigned_region_count": 2,
            "assigned_review_region_count": 0, "unassigned_region_count": 0, "structural_region_count": 0,
            "review_region_count": 0,
            "records": [
                {"region_id": "region_policy-line", "region_type": "INPUT_LINE", "bbox_px": [10, 30, 90, 50],
                 "parent_region_id": None, "status": "ASSIGNED", "field_id": "field_policy", "semantic_key": "policy",
                 "field_type": "text", "confidence": 0.92, "needs_review": False, "assignment_review_required": False},
                {"region_id": "region_accept-check", "region_type": "CHECKBOX", "bbox_px": [10, 70, 30, 90],
                 "parent_region_id": None, "status": "ASSIGNED", "field_id": "field_confirmed", "semantic_key": "confirmed",
                 "field_type": "boolean", "confidence": 0.88, "needs_review": False, "assignment_review_required": False},
            ],
        },
        "table_output": {"schema_version": "1.0.0", "document_id": "visual-mock-document", "page_id": "page_001", "table_count": 0, "tables": []},
        "consistency_warnings": [],
        "quality_summary": {
            "target_region_count": 2, "semantic_field_count": 2, "assigned_region_count": 2,
            "assigned_review_region_count": 0, "unassigned_region_count": 0, "structural_region_count": 0,
            "actionable_coverage_ratio": 1.0, "semantic_consistency_warning_count": 0,
            "structured_table_count": 0, "mapping_complete": True, "quality_status": "MAPPED", "automation_ready": False,
        },
    }


@app.post("/api/v1/templates/register", status_code=201)
def register_definition(payload: dict[str, Any]):
    return payload


@app.post("/api/v1/templates/{template_id}/reference", status_code=201)
async def register_template_reference(template_id: str, file: UploadFile = File(...)):
    await file.read()
    return {"template_id": template_id, "stored": True}


@app.post("/api/v1/documents/process")
async def process_document(file: UploadFile = File(...), template_id: str = Form(...)):
    await file.read()
    return {"job_id": "mock-document-job", "template_id": template_id, "status": "HUMAN_REVIEW_REQUIRED",
            "quality_check": {"is_passed": True}, "overall_confidence": 0.8, "needs_human_review": True,
            "extracted_fields": [{"field_id": "field_policy", "label": "Policy", "field_type": "handwriting", "raw_text": "MTR 001", "normalized_text": "MTR001", "ocr_confidence": 0.8, "validation_passed": True, "validation_message": None, "final_confidence": 0.8, "human_review_flag": True, "crop_image_path": "crop.png"}]}


@app.get("/api/v1/documents/jobs/{job_id}/export/{export_format}")
def export(job_id: str, export_format: str):
    return Response(b'{"job_id":"mock-document-job"}', media_type="application/json")
