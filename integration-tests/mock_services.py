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
    return {"document_id": document_id, "status": "preprocessed", "summary": {"page_count": 1}}


@app.post("/v1/documents/{document_id}/extract")
def extract(document_id: str, payload: dict[str, Any]):
    return {"document_id": document_id, "status": "extracted", "summary": {"region_count": 2}}


@app.get("/v1/documents/{document_id}/result")
def layout_result(document_id: str):
    return {
        "document_id": document_id,
        "model": {"name": "PP-DocLayoutV3-mock", "version": "1.0"},
        "pages": [
            {
                "page_id": "page_001", "page_number": 1, "width": 100, "height": 200,
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
    return Response(PAGE, media_type="image/png")


@app.post("/v1/ocr/process")
async def ocr(file: UploadFile = File(...)):
    content = await file.read()
    return {
        "schema_version": "1.0.0", "document_id": "ocr-mock", "model": {"engine": "mock-ocr", "version": "1"},
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
    return {"job_id": "job_mock_vlm", "status": "COMPLETED", "accepted": True, "review_required": True,
            "semantic_output": {"pages": [{"page_id": "page_001", "semantic_labels": [], "fields": []}], "warnings": [{"code": "MOCK_ENGINE", "message": "Human mapping required", "page_id": "page_001", "severity": "WARNING"}]},
            "quality_summary": {"actionable_coverage_ratio": 0, "quality_status": "INCOMPLETE_REVIEW_REQUIRED"}}


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
