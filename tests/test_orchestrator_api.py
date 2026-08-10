from __future__ import annotations

import io
import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from orchestrator.config import Settings
from orchestrator.database import create_session_factory
from orchestrator.main import create_app


def png_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (100, 200), "white").save(output, format="PNG")
    return output.getvalue()


def test_plain_postgresql_url_uses_installed_psycopg3_driver():
    engine, _ = create_session_factory("postgresql://user:password@db/example")
    try:
        assert engine.url.drivername == "postgresql+psycopg"
    finally:
        engine.dispose()


class FakeDownstreams:
    def __init__(self):
        self.page = png_bytes()
        self.registered_templates = []
        self.processed_documents = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        method = request.method
        if path in {"/health", "/health/ready"}:
            return httpx.Response(200, json={"status": "ok"})
        if method == "POST" and path == "/v1/documents":
            return httpx.Response(201, json={"document_id": "visual-doc", "status": "uploaded"})
        if method == "POST" and path.endswith("/preprocess"):
            return httpx.Response(200, json={"status": "preprocessed"})
        if method == "POST" and path.endswith("/extract"):
            return httpx.Response(200, json={"status": "extracted"})
        if method == "GET" and path.endswith("/result") and path.startswith("/v1/documents"):
            return httpx.Response(
                200,
                json={
                    "document_id": "visual-doc",
                    "model": {"name": "PP-DocLayoutV3", "version": "test"},
                    "pages": [
                        {
                            "page_id": "page_001",
                            "page_number": 1,
                            "width": 100,
                            "height": 200,
                            "image_path": "preprocessed/pages/page_001.png",
                            "regions": [
                                {"region_id": "line-1", "class_name": "input_line", "confidence": 0.9, "bbox_px": [10, 30, 90, 50]},
                                {"region_id": "check-1", "class_name": "checkbox", "confidence": 0.9, "bbox_px": [10, 70, 30, 90]},
                            ],
                        }
                    ],
                },
            )
        if method == "GET" and "/artifacts/" in path:
            return httpx.Response(200, content=self.page, headers={"content-type": "image/png"})
        if method == "POST" and path == "/v1/ocr/process":
            return httpx.Response(
                200,
                json={
                    "model": {"engine": "Tesseract", "version": "1"},
                    "pages": [
                        {
                            "tokens": [
                                {"token_id": "tok_0001", "text": "Policy", "normalized_text": "Policy", "language": "eng", "confidence": 0.9, "reading_order": 0, "bounding_box": [10, 10, 40, 20]}
                            ]
                        }
                    ],
                },
            )
        if method == "POST" and path == "/api/v1/registrations":
            return httpx.Response(202, json={"job_id": "job_vlm", "status": "PENDING"})
        if method == "GET" and path == "/api/v1/registrations/job_vlm":
            return httpx.Response(200, json={"job_id": "job_vlm", "status": "COMPLETED"})
        if method == "GET" and path == "/api/v1/registrations/job_vlm/result":
            return httpx.Response(
                200,
                json={
                    "status": "COMPLETED",
                    "review_required": True,
                    "semantic_output": {"pages": [{"page_id": "page_001", "semantic_labels": [], "fields": []}], "warnings": []},
                    "quality_summary": {"actionable_coverage_ratio": 0, "quality_status": "INCOMPLETE_REVIEW_REQUIRED"},
                },
            )
        if method == "POST" and path == "/api/v1/templates/register":
            definition = json.loads(request.content)
            self.registered_templates.append(definition)
            return httpx.Response(201, json=definition)
        if method == "POST" and path == "/api/v1/documents/process":
            self.processed_documents += 1
            return httpx.Response(
                200,
                json={
                    "job_id": "downstream-job",
                    "template_id": self.registered_templates[-1]["template_id"],
                    "status": "HUMAN_REVIEW_REQUIRED",
                    "extracted_fields": [
                        {
                            "field_id": "field_policy",
                            "label": "Policy",
                            "field_type": "handwriting",
                            "raw_text": "MTR 123",
                            "normalized_text": "MTR123",
                            "ocr_confidence": 0.75,
                            "validation_passed": True,
                            "validation_message": None,
                            "final_confidence": 0.75,
                            "human_review_flag": True,
                            "crop_image_path": "crop.png",
                        }
                    ],
                    "overall_confidence": 0.75,
                    "needs_human_review": True,
                    "quality_check": {"is_passed": True},
                },
            )
        if method == "GET" and "/export/" in path:
            return httpx.Response(200, content=b"export", headers={"content-type": "application/octet-stream"})
        return httpx.Response(404, json={"detail": f"unhandled {method} {path}"})


@pytest.fixture()
def client(tmp_path: Path):
    fake = FakeDownstreams()
    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'test.db'}",
        storage_root=tmp_path / "artifacts",
        document_processing_url="http://downstream",
        visual_field_url="http://downstream",
        ocr_url="http://downstream",
        vlm_url="http://downstream",
        vlm_api_key="test-key",
        request_timeout_seconds=2,
        retry_attempts=1,
        retry_backoff_seconds=0,
        poll_interval_seconds=0.001,
        poll_timeout_seconds=2,
        max_upload_mb=2,
        allowed_extensions=(".png", ".pdf"),
        cors_origins=("http://localhost:3000",),
    )
    app = create_app(settings, downstream_transport=httpx.MockTransport(fake))
    with TestClient(app) as test_client:
        yield test_client, fake


def register_and_approve(client: TestClient):
    response = client.post(
        "/api/template-registrations?form_type_id=motor",
        files=[("files", ("blank.png", png_bytes(), "image/png"))],
    )
    assert response.status_code == 202
    registration_id = response.json()["items"][0]["id"]
    registration = client.get(f"/api/v1/template-registrations/{registration_id}").json()
    assert registration["status"] == "needs_approval"
    assert registration["image_identity"]["width"] == 100
    approved = client.post(f"/api/template-registrations/{registration_id}/approve")
    assert approved.status_code == 200, approved.text
    return registration_id, approved.json()["template"]


def test_compatibility_and_canonical_template_routes_share_record(client):
    test_client, fake = client
    registration_id, template = register_and_approve(test_client)
    canonical = test_client.get(f"/api/v1/template-registrations/{registration_id}").json()
    compatibility = test_client.get("/api/template-registrations").json()
    assert canonical["template_id"] == template["id"]
    assert next(item for item in compatibility if item["id"] == registration_id)["template_id"] == template["id"]
    assert len(fake.registered_templates) == 1


def test_stale_draft_revision_returns_409(client):
    test_client, _ = client
    response = test_client.post(
        "/api/template-registrations?form_type_id=motor",
        files=[("files", ("blank.png", png_bytes(), "image/png"))],
    )
    registration_id = response.json()["items"][0]["id"]
    registration = test_client.get(f"/api/v1/template-registrations/{registration_id}").json()
    stale = test_client.put(
        f"/api/v1/template-registrations/{registration_id}/draft",
        json={"revision": 0, "regions": registration["draft"]["regions"]},
    )
    assert stale.status_code == 409


def test_document_workflow_review_approval_and_real_exports(client):
    test_client, fake = client
    _, template = register_and_approve(test_client)
    # Simulate a document-processing container restart, which clears its
    # current in-memory template registry.
    fake.registered_templates.clear()
    upload = test_client.post(
        f"/api/documents?template_id={template['id']}&process_immediately=true",
        files=[("files", ("filled.png", png_bytes(), "image/png"))],
    )
    assert upload.status_code == 202
    document_id = upload.json()["items"][0]["id"]
    canonical = test_client.get(f"/api/v1/documents/{document_id}").json()
    assert canonical["status"] == "needs_review"
    assert canonical["processed"]["fields"]["field_policy"]["raw_value"] == "MTR 123"
    review = test_client.put(
        f"/api/v1/documents/{document_id}/review",
        json={"reviewer": "alice", "fields": [{"field_id": "field_policy", "corrected_value": "MTR124", "reason": "scan correction"}]},
    )
    assert review.status_code == 200
    approved = test_client.post(f"/api/v1/documents/{document_id}/approve")
    assert approved.status_code == 200
    export = test_client.get("/api/export/csv")
    assert export.status_code == 200
    assert "MTR124" in export.text
    upstream_export = test_client.get(f"/api/v1/documents/{document_id}/export/json")
    assert upstream_export.content == b"export"
    assert fake.processed_documents == 1
    assert len(fake.registered_templates) == 1


def test_invalid_ids_and_unsupported_uploads(client):
    test_client, _ = client
    assert test_client.get("/api/v1/documents/../../etc/passwd").status_code == 404
    response = test_client.post(
        "/api/template-registrations?form_type_id=motor",
        files=[("files", ("bad.exe", b"not an image", "application/octet-stream"))],
    )
    assert response.status_code == 415


def test_cors_same_origin_configuration(client):
    test_client, _ = client
    response = test_client.options(
        "/api/templates",
        headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "GET"},
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
