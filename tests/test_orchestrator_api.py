from __future__ import annotations

import hashlib
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
        self.registered_references = []
        self.processed_documents = 0
        self.retake_required = False
        self.quality_pass = True
        self.preprocessing_decision = "canonicalize_only"
        self.preprocessing_operations = ["exif_orientation", "rgb_conversion"]
        self.fail_layout = False
        self.fail_ocr = False
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        method = request.method
        if path in {"/health", "/health/ready"}:
            return httpx.Response(200, json={"status": "ok"})
        if method == "POST" and path == "/v1/documents":
            return httpx.Response(201, json={"document_id": "visual-doc", "status": "uploaded"})
        if method == "POST" and path.endswith("/preprocess"):
            capture_ready = not self.retake_required and self.quality_pass
            decision = "reject_and_retake" if not capture_ready else self.preprocessing_decision
            reasons = (
                ["image_too_blurry"]
                if self.retake_required
                else (["image_may_be_blurry"] if not self.quality_pass else [])
            )
            return httpx.Response(
                200,
                json={
                    "document_id": "visual-doc",
                    "status": "preprocessed",
                    "artifact": "preprocessed/preprocess_manifest.json",
                    "summary": {
                        "page_count": 1,
                        "quality_pass": self.quality_pass,
                        "capture_profile": "template",
                        "decision": decision,
                        "capture_ready": capture_ready,
                        "retake_required": not capture_ready,
                        "reasons": reasons,
                        "advisories": [],
                        "instructions": ["Hold still, tap to focus and capture again."] if self.retake_required else [],
                        "operations": self.preprocessing_operations,
                    },
                },
            )
        if method == "GET" and path.endswith("/capture-assessment"):
            capture_ready = not self.retake_required
            reasons = ["image_too_blurry"] if self.retake_required else []
            instructions = ["Hold still, tap to focus and capture again."] if self.retake_required else []
            return httpx.Response(
                200,
                json={
                    "document_id": "visual-doc",
                    "capture_profile": "template",
                    "capture_ready": capture_ready,
                    "retake_required": self.retake_required,
                    "pages": [
                        {
                            "page_id": "page_001",
                            "status": "retake_required" if self.retake_required else "ready",
                            "capture_ready": capture_ready,
                            "retake_required": self.retake_required,
                            "reasons": reasons,
                            "advisories": [],
                            "instructions": instructions,
                            "alignment": None,
                        }
                    ],
                },
            )
        if method == "GET" and path.endswith("/artifacts/preprocessed/preprocess_manifest.json"):
            capture_ready = not self.retake_required and self.quality_pass
            decision = "reject_and_retake" if not capture_ready else self.preprocessing_decision
            reasons = (
                ["image_too_blurry"]
                if self.retake_required
                else (["image_may_be_blurry"] if not self.quality_pass else [])
            )
            return httpx.Response(
                200,
                json={
                    "schema_version": "1.3.0",
                    "capture_profile": "template",
                    "decision": decision,
                    "capture_ready": capture_ready,
                    "retake_required": not capture_ready,
                    "reasons": reasons,
                    "advisories": [],
                    "instructions": ["Hold still, tap to focus and capture again."] if self.retake_required else [],
                    "operations": self.preprocessing_operations,
                    "page_count": 1,
                    "quality_pass": self.quality_pass,
                    "pages": [
                        {
                            "image_path": "preprocessed/pages/page_001.png",
                            "page_id": "page_001",
                            "page_number": 1,
                            "width": 100,
                            "height": 200,
                            "sha256": hashlib.sha256(self.page).hexdigest(),
                            "source_to_page_transform": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                            "operations": self.preprocessing_operations,
                            "quality": {
                                "quality_pass": self.quality_pass and capture_ready,
                                "warnings": ["image_may_be_blurry"] if not self.quality_pass or self.retake_required else [],
                            },
                        }
                    ],
                },
            )
        if method == "POST" and path.endswith("/extract"):
            if self.fail_layout:
                return httpx.Response(503, json={"detail": "layout unavailable"})
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
            if self.fail_ocr:
                return httpx.Response(503, json={"detail": "ocr unavailable"})
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
        if method == "POST" and path.startswith("/api/v1/templates/") and path.endswith("/reference"):
            self.registered_references.append(path)
            return httpx.Response(201, json={"stored": True})
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

@pytest.mark.parametrize(
    ("policy", "correction_mode", "operations", "decision"),
    [
        ("none", "none", ["exif_orientation", "rgb_conversion"], "canonicalize_only"),
        ("force", "standard", ["exif_orientation", "rgb_conversion", "mild_sharpening"], "correct"),
    ],
)
def test_preprocessing_policy_is_mapped_and_decision_is_persisted(
    client,
    policy,
    correction_mode,
    operations,
    decision,
):
    test_client, fake = client
    fake.preprocessing_decision = decision
    fake.preprocessing_operations = operations
    page = png_bytes()

    response = test_client.post(
        "/api/v1/template-registrations",
        data={"form_type_id": "motor", "preprocessing_policy": policy},
        files={"file": ("blank.png", page, "image/png")},
    )

    assert response.status_code == 202, response.text
    registration = test_client.get(
        f"/api/v1/template-registrations/{response.json()['id']}"
    ).json()
    assert registration["status"] == "needs_approval"
    assert registration["source_sha256"] == hashlib.sha256(page).hexdigest()
    assert registration["preprocessing"]["requested_policy"] == policy
    assert registration["preprocessing"]["correction_mode"] == correction_mode
    assert registration["preprocessing"]["decision"] == decision
    assert not any(
        request.url.path.endswith("/capture-assessment")
        for request in fake.requests
    )
    preprocess_request = next(
        request
        for request in fake.requests
        if request.method == "POST" and request.url.path.endswith("/preprocess")
    )
    assert json.loads(preprocess_request.content) == {
        "correction_mode": correction_mode,
        "capture_profile": "template",
        "deskew": True,
        "normalize_illumination": True,
        "sharpen": True,
    }


def test_failed_capture_stops_before_layout_ocr_and_vlm(client):
    test_client, fake = client
    fake.retake_required = True

    response = test_client.post(
        "/api/v1/template-registrations",
        data={"form_type_id": "motor", "preprocessing_policy": "auto"},
        files={"file": ("blurry.png", png_bytes(), "image/png")},
    )

    assert response.status_code == 202, response.text
    registration = test_client.get(
        f"/api/v1/template-registrations/{response.json()['id']}"
    ).json()
    assert registration["status"] == "needs_resubmission"
    assert registration["progress"]["stage"] == "capture_quality"
    assert registration["failure"] is None
    assert registration["preprocessing"]["decision"] == "reject_and_retake"
    assert registration["preprocessing"]["capture_ready"] is False
    assert registration["preprocessing"]["retake_required"] is True
    assert "image_too_blurry" in registration["preprocessing"]["reasons"]
    assert registration["preprocessing"]["instructions"]

    attempted = {(request.method, request.url.path) for request in fake.requests}
    assert ("POST", "/v1/documents/visual-doc/extract") not in attempted
    assert ("POST", "/v1/ocr/process") not in attempted
    assert ("POST", "/api/v1/registrations") not in attempted


def test_missing_authoritative_preprocessing_decision_fails_contract(client):
    test_client, fake = client
    fake.preprocessing_decision = None

    response = test_client.post(
        "/api/v1/template-registrations",
        data={"form_type_id": "motor", "preprocessing_policy": "auto"},
        files={"file": ("blank.png", png_bytes(), "image/png")},
    )

    assert response.status_code == 202, response.text
    registration = test_client.get(
        f"/api/v1/template-registrations/{response.json()['id']}"
    ).json()
    assert registration["status"] == "failed"
    assert registration["failure"] == {
        "code": "CONTRACT_ERROR",
        "message": "Preprocessing manifest did not provide a valid authoritative decision",
    }
    attempted = {(request.method, request.url.path) for request in fake.requests}
    assert ("POST", "/v1/documents/visual-doc/extract") not in attempted
    assert ("POST", "/v1/ocr/process") not in attempted
    assert ("POST", "/api/v1/registrations") not in attempted


def test_invalid_preprocessing_policy_is_rejected(client):
    test_client, fake = client

    response = test_client.post(
        "/api/v1/template-registrations",
        data={"form_type_id": "motor", "preprocessing_policy": "aggressive"},
        files={"file": ("blank.png", png_bytes(), "image/png")},
    )

    assert response.status_code == 422
    assert "preprocessing_policy must be one of" in response.json()["detail"]
    assert fake.requests == []

def test_none_policy_still_blocks_manifest_quality_failure(client):
    test_client, fake = client
    fake.quality_pass = False

    response = test_client.post(
        "/api/v1/template-registrations",
        data={"form_type_id": "motor", "preprocessing_policy": "none"},
        files={"file": ("blurry-scanner.png", png_bytes(), "image/png")},
    )

    assert response.status_code == 202, response.text
    registration = test_client.get(
        f"/api/v1/template-registrations/{response.json()['id']}"
    ).json()
    preprocessing = registration["preprocessing"]
    assert registration["status"] == "needs_resubmission"
    assert preprocessing["requested_policy"] == "none"
    assert preprocessing["quality_pass"] is False
    assert preprocessing["capture_ready"] is False
    assert preprocessing["retake_required"] is True
    assert "image_may_be_blurry" in preprocessing["reasons"]

    attempted = {(request.method, request.url.path) for request in fake.requests}
    assert ("POST", "/v1/documents/visual-doc/extract") not in attempted
    assert ("POST", "/v1/ocr/process") not in attempted
    assert ("POST", "/api/v1/registrations") not in attempted


def test_canonical_page_is_established_before_layout_and_ocr(client):
    test_client, fake = client

    response = test_client.post(
        "/api/v1/template-registrations",
        data={"form_type_id": "motor", "preprocessing_policy": "auto"},
        files={"file": ("blank.png", fake.page, "image/png")},
    )

    assert response.status_code == 202, response.text
    registration_id = response.json()["id"]
    registration = test_client.get(
        f"/api/v1/template-registrations/{registration_id}"
    ).json()
    expected_sha = hashlib.sha256(fake.page).hexdigest()
    assert registration["status"] == "needs_approval"
    assert registration["layout_status"] == "complete"
    assert registration["ocr_status"] == "complete"
    assert registration["image_identity"] == {
        "sha256": expected_sha,
        "width": 100,
        "height": 200,
        "document_id": "visual-doc",
        "page_id": "page_001",
        "page_number": 1,
    }
    assert registration["preprocessing"]["pages"][0]["image_path"] == (
        "preprocessed/pages/page_001.png"
    )
    page_response = test_client.get(
        f"/api/v1/template-registrations/{registration_id}/pages/1"
    )
    assert page_response.status_code == 200
    assert page_response.content == fake.page

    requests = [
        (request.method, request.url.path)
        for request in fake.requests
    ]
    canonical_index = requests.index(
        ("GET", "/v1/documents/visual-doc/artifacts/preprocessed/pages/page_001.png")
    )
    layout_index = requests.index(("POST", "/v1/documents/visual-doc/extract"))
    ocr_index = requests.index(("POST", "/v1/ocr/process"))
    assert canonical_index < layout_index
    assert canonical_index < ocr_index


@pytest.mark.parametrize(
    ("failed_branch", "failure_service"),
    [
        ("layout", "visual-field-detection"),
        ("ocr", "ocr-fastapi-service"),
    ],
)
def test_branch_failure_is_observable_and_prevents_vlm(
    client,
    failed_branch,
    failure_service,
):
    test_client, fake = client
    setattr(fake, f"fail_{failed_branch}", True)

    response = test_client.post(
        "/api/v1/template-registrations",
        data={"form_type_id": "motor"},
        files={"file": ("blank.png", fake.page, "image/png")},
    )

    assert response.status_code == 202, response.text
    registration = test_client.get(
        f"/api/v1/template-registrations/{response.json()['id']}"
    ).json()
    other_branch = "ocr" if failed_branch == "layout" else "layout"
    assert registration["status"] == "failed"
    assert registration[f"{failed_branch}_status"] == "failed"
    assert registration[f"{other_branch}_status"] == "complete"
    assert registration["failure"]["code"] == "DOWNSTREAM_ERROR"
    assert registration["failure"]["service"] == failure_service
    attempted = {(request.method, request.url.path) for request in fake.requests}
    assert ("POST", "/api/v1/registrations") not in attempted
