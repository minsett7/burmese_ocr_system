from __future__ import annotations

import csv
import io
import json
import logging
import re
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, Body, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from openpyxl import Workbook

from adapters.contracts import AdapterError

from . import __version__
from .config import Settings
from .database import create_session_factory
from .downstream import DownstreamClient, DownstreamError
from .store import RecordStore, iso_now
from .workflows import WorkflowService


logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","message":"%(message)s"}',
)
logger = logging.getLogger("orchestrator")

FORM_TYPES = [
    {"id": "health", "name": "Health", "label": "Health Claim"},
    {"id": "life", "name": "Life", "label": "Life Claim"},
    {"id": "motor", "name": "Motor", "label": "Motor Claim"},
    {"id": "fire", "name": "Fire", "label": "Fire Claim"},
]
VALID_FORM_TYPES = {item["id"] for item in FORM_TYPES}
ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,99}$")


def _id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12].upper()}"


def _record_or_404(store: RecordStore, kind: str, record_id: str) -> dict[str, Any]:
    if not ID_PATTERN.fullmatch(record_id):
        raise HTTPException(status_code=404, detail=f"{kind} not found")
    value = store.get(kind, record_id)
    if value is None or value.get("deleted_at"):
        raise HTTPException(status_code=404, detail=f"{kind} not found")
    return value


async def _read_upload(upload: UploadFile, settings: Settings) -> bytes:
    filename = upload.filename or ""
    suffix = Path(filename).suffix.lower()
    if suffix not in settings.allowed_extensions:
        raise HTTPException(status_code=415, detail=f"unsupported upload type: {suffix or 'none'}")
    chunks: list[bytes] = []
    size = 0
    while chunk := await upload.read(1024 * 1024):
        size += len(chunk)
        if size > settings.max_upload_bytes:
            raise HTTPException(status_code=413, detail="uploaded file is too large")
        chunks.append(chunk)
    if not chunks:
        raise HTTPException(status_code=400, detail="uploaded file is empty")
    return b"".join(chunks)


def create_app(
    settings: Settings | None = None,
    *,
    downstream_transport=None,
) -> FastAPI:
    configured = settings or Settings.from_env()
    engine, sessions = create_session_factory(configured.database_url)
    store = RecordStore(engine, sessions)
    client = DownstreamClient(
        configured.request_timeout_seconds,
        configured.retry_attempts,
        configured.retry_backoff_seconds,
        transport=downstream_transport,
    )
    workflows = WorkflowService(configured, store, client)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        store.initialize()
        yield
        engine.dispose()

    app = FastAPI(
        title="Unified Burmese Insurance Platform",
        version=__version__,
        description="Thin orchestration and durable review API across the independent layout, OCR, VLM, and document-processing services.",
        lifespan=lifespan,
    )
    app.state.settings = configured
    app.state.store = store
    app.state.client = client
    app.state.workflows = workflows

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(configured.cors_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Correlation-ID", "Content-Disposition"],
    )

    @app.middleware("http")
    async def correlation_middleware(request: Request, call_next):
        supplied = request.headers.get("X-Correlation-ID", "")
        correlation_id = supplied if re.fullmatch(r"[A-Za-z0-9_.:-]{1,100}", supplied) else str(uuid.uuid4())
        request.state.correlation_id = correlation_id
        response = await call_next(request)
        response.headers["X-Correlation-ID"] = correlation_id
        logger.info(
            "%s %s %s",
            request.method,
            request.url.path,
            response.status_code,
            extra={"correlation_id": correlation_id},
        )
        return response

    @app.exception_handler(DownstreamError)
    async def downstream_handler(request: Request, exc: DownstreamError):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": "DOWNSTREAM_ERROR",
                    "message": exc.message,
                    "service": exc.service,
                    "upstream_status": exc.upstream_status,
                    "details": exc.detail,
                    "correlation_id": request.state.correlation_id,
                }
            },
        )

    @app.exception_handler(AdapterError)
    async def adapter_handler(request: Request, exc: AdapterError):
        return JSONResponse(
            status_code=422,
            content={"error": {"code": "CONTRACT_ERROR", "message": str(exc), "correlation_id": request.state.correlation_id}},
        )

    @app.get("/health", tags=["health"])
    async def health() -> dict[str, Any]:
        return {"status": "ok", "version": __version__}

    @app.get("/ready", tags=["health"])
    async def ready() -> dict[str, Any]:
        try:
            with sessions() as session:
                session.connection().exec_driver_sql("SELECT 1")
        except Exception as exc:
            raise HTTPException(status_code=503, detail="PostgreSQL is not ready") from exc
        return {"status": "ready", "database": "ready"}

    @app.get("/api/v1/services/status", tags=["health"])
    async def services_status(request: Request) -> dict[str, Any]:
        targets = {
            "document_processing": (configured.document_processing_url, "/health"),
            "visual_field_detection": (configured.visual_field_url, "/health"),
            "ocr": (configured.ocr_url, "/health"),
            "insurance_vlm": (configured.vlm_url, "/health/ready"),
        }
        results: dict[str, Any] = {}
        for name, (base, path) in targets.items():
            headers = {"X-API-Key": configured.vlm_api_key} if name == "insurance_vlm" and configured.vlm_api_key else {}
            try:
                response = await client.request(name, "GET", base + path, correlation_id=request.state.correlation_id, headers=headers)
                results[name] = {"status": "available", "details": response.json()}
            except DownstreamError as exc:
                results[name] = {"status": "unavailable", "error": exc.message}
        return {"status": "ok" if all(item["status"] == "available" for item in results.values()) else "degraded", "services": results}

    def create_registration_record(
        *, content: bytes, filename: str, name: str, form_type_id: str, language: str, version_note: str | None, correlation_id: str
    ) -> dict[str, Any]:
        if form_type_id not in VALID_FORM_TYPES:
            raise HTTPException(status_code=422, detail="unknown form_type_id")
        registration_id = _id("REG")
        source_path = workflows.save_upload(registration_id, filename, content)
        now = iso_now()
        record = {
            "id": registration_id,
            "template_id": None,
            "name": name,
            "form_type_id": form_type_id,
            "language": language,
            "version_note": version_note,
            "file_name": filename,
            "source_path": str(source_path),
            "status": "analyzing",
            "progress": {"stage": "queued", "percent": 0},
            "draft": None,
            "draft_revision": 0,
            "downstream_ids": {},
            "failure": None,
            "correlation_id": correlation_id,
            "created_at": now,
            "updated_at": now,
            "approved_at": None,
        }
        store.put("registration", registration_id, record, create_only=True)
        store.add_audit(
            action="created template registration",
            target_type="template_registration",
            target_id=registration_id,
            correlation_id=correlation_id,
            after={"file_name": filename, "form_type_id": form_type_id},
        )
        return record

    @app.post("/api/v1/template-registrations", status_code=202, tags=["templates"])
    async def create_registration(
        request: Request,
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        name: str = Form("Insurance Claim Template"),
        form_type_id: str = Form("motor"),
        language: str = Form("my-en"),
        version_note: str | None = Form(None),
    ) -> dict[str, Any]:
        content = await _read_upload(file, configured)
        record = create_registration_record(
            content=content,
            filename=file.filename or "template",
            name=name,
            form_type_id=form_type_id,
            language=language,
            version_note=version_note,
            correlation_id=request.state.correlation_id,
        )
        background_tasks.add_task(workflows.run_template_registration, record["id"])
        return {"id": record["id"], "job_id": record["id"], "status": record["status"], "progress": record["progress"]}

    @app.post("/api/v1/templates/register", status_code=202, tags=["templates"])
    async def legacy_register_template(
        request: Request,
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        name: str = Form("Insurance Claim Template"),
        form_type_id: str = Form("motor"),
    ) -> dict[str, Any]:
        content = await _read_upload(file, configured)
        record = create_registration_record(
            content=content, filename=file.filename or "template", name=name, form_type_id=form_type_id,
            language="my-en", version_note=None, correlation_id=request.state.correlation_id,
        )
        background_tasks.add_task(workflows.run_template_registration, record["id"])
        return {"job_id": record["id"], "status": record["status"]}

    @app.get("/api/v1/template-registrations/{registration_id}", tags=["templates"])
    @app.get("/api/v1/templates/jobs/{registration_id}", tags=["templates"])
    async def get_registration(registration_id: str) -> dict[str, Any]:
        return _record_or_404(store, "registration", registration_id)

    @app.get("/api/v1/templates/jobs/{registration_id}/result", tags=["templates"])
    async def get_registration_result(registration_id: str) -> dict[str, Any]:
        record = _record_or_404(store, "registration", registration_id)
        if record["status"] == "failed":
            raise HTTPException(status_code=422, detail=record["failure"])
        if not record.get("draft"):
            raise HTTPException(status_code=409, detail={"status": record["status"]})
        return {"job_id": registration_id, "status": record["status"], "draft": record["draft"], "vlm_result": record.get("vlm_result")}

    @app.get("/api/v1/template-registrations/{registration_id}/pages/{page_number}", tags=["templates"])
    async def registration_page(registration_id: str, page_number: int):
        _record_or_404(store, "registration", registration_id)
        if page_number != 1:
            raise HTTPException(status_code=404, detail="page not found")
        path = configured.storage_root / registration_id / "page_001.png"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="page not found")
        return FileResponse(path, media_type="image/png")

    @app.put("/api/v1/template-registrations/{registration_id}/draft", tags=["templates"])
    async def save_draft(registration_id: str, request: Request, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        record = _record_or_404(store, "registration", registration_id)
        supplied_revision = payload.get("revision")
        if supplied_revision != record.get("draft_revision"):
            raise HTTPException(status_code=409, detail={"code": "STALE_REVISION", "current_revision": record.get("draft_revision")})
        regions = payload.get("regions")
        if not isinstance(regions, list):
            raise HTTPException(status_code=422, detail="regions must be a list")
        before = record.get("draft")
        new_revision = int(record["draft_revision"]) + 1
        record["draft"] = {**(record.get("draft") or {}), **payload, "revision": new_revision, "regions": regions}
        record["draft_revision"] = new_revision
        record["updated_at"] = iso_now()
        store.put("registration", registration_id, record)
        store.add_audit(
            action="updated template draft", target_type="template_registration", target_id=registration_id,
            actor=request.headers.get("X-Actor", "reviewer"), before={"revision": supplied_revision},
            after={"revision": new_revision}, correlation_id=request.state.correlation_id,
        )
        return record

    def validate_registration_record(record: dict[str, Any]) -> list[str]:
        errors: list[str] = []
        draft = record.get("draft") or {}
        if not draft.get("regions"):
            errors.append("draft has no detected fields")
        keys: set[str] = set()
        for region in draft.get("regions", []):
            key = str(region.get("key", ""))
            if not key:
                errors.append(f"{region.get('id')}: key is required")
            elif key in keys:
                errors.append(f"duplicate field key: {key}")
            keys.add(key)
            if not region.get("bbox"):
                errors.append(f"{region.get('id')}: geometry is required")
            if not region.get("extraction_mode"):
                errors.append(f"{region.get('id')}: unsupported or ambiguous field type must be resolved")
        return errors

    @app.post("/api/v1/template-registrations/{registration_id}/validate", tags=["templates"])
    async def validate_registration(registration_id: str) -> dict[str, Any]:
        record = _record_or_404(store, "registration", registration_id)
        errors = validate_registration_record(record)
        return {"valid": not errors, "errors": errors, "revision": record.get("draft_revision")}

    async def approve_registration_common(registration_id: str, request: Request) -> dict[str, Any]:
        record = _record_or_404(store, "registration", registration_id)
        errors = validate_registration_record(record)
        if errors:
            raise HTTPException(status_code=422, detail={"code": "DRAFT_INVALID", "errors": errors})
        try:
            return await workflows.approve_template(
                registration_id,
                request.headers.get("X-Actor", "reviewer"),
                request.state.correlation_id,
            )
        except (ValueError, KeyError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/v1/template-registrations/{registration_id}/approve", tags=["templates"])
    @app.post("/api/v1/templates/jobs/{registration_id}/approve", tags=["templates"])
    async def approve_registration(registration_id: str, request: Request) -> dict[str, Any]:
        return await approve_registration_common(registration_id, request)

    @app.get("/api/form-types", tags=["compatibility"])
    async def form_types() -> list[dict[str, str]]:
        return FORM_TYPES

    @app.get("/api/templates", tags=["compatibility"])
    async def list_templates() -> list[dict[str, Any]]:
        return [item for item in store.list("template") if not item.get("deleted_at")]

    @app.get("/api/templates/{template_id}", tags=["compatibility"])
    async def get_template(template_id: str) -> dict[str, Any]:
        return _record_or_404(store, "template", template_id)

    @app.get("/api/template-registrations", tags=["compatibility"])
    async def list_registrations() -> list[dict[str, Any]]:
        result = []
        for item in store.list("registration"):
            if item.get("deleted_at"):
                continue
            result.append({
                **item,
                "fields": [region.get("key") for region in (item.get("draft") or {}).get("regions", [])],
                "quality_score": (item.get("vlm_result") or {}).get("quality_summary", {}).get("actionable_coverage_ratio", 0),
                "layout_score": 1 if item.get("layout_contract") else 0,
                "detected_regions": len((item.get("draft") or {}).get("regions", [])),
            })
        return result

    @app.post("/api/template-registrations", status_code=202, tags=["compatibility"])
    async def compatibility_create_registrations(
        request: Request,
        background_tasks: BackgroundTasks,
        form_type_id: str = Query(...),
        files: list[UploadFile] = File(...),
    ) -> dict[str, Any]:
        items = []
        for upload in files:
            content = await _read_upload(upload, configured)
            item = create_registration_record(
                content=content, filename=upload.filename or "template", name=f"{form_type_id.title()} Claim Template",
                form_type_id=form_type_id, language="my-en", version_note=None, correlation_id=request.state.correlation_id,
            )
            background_tasks.add_task(workflows.run_template_registration, item["id"])
            items.append(item)
        return {"items": items}

    @app.patch("/api/template-registrations/{registration_id}/fields", tags=["compatibility"])
    async def compatibility_update_fields(registration_id: str, request: Request, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        fields = payload.get("fields")
        if not isinstance(fields, list):
            raise HTTPException(status_code=400, detail="payload.fields must be a list")
        record = _record_or_404(store, "registration", registration_id)
        draft = record.get("draft") or {"regions": []}
        if len(fields) != len(draft.get("regions", [])):
            raise HTTPException(status_code=422, detail="field list must match the detected region count; use the canonical draft endpoint for geometry edits")
        updated_regions = []
        for region, key in zip(draft["regions"], fields):
            updated_regions.append({**region, "key": str(key), "label": str(key).replace("_", " ").title()})
        record["draft"] = {**draft, "regions": updated_regions, "revision": record["draft_revision"] + 1}
        record["draft_revision"] += 1
        record["updated_at"] = iso_now()
        store.put("registration", registration_id, record)
        store.add_audit(
            action="updated template field map", target_type="template_registration", target_id=registration_id,
            actor=request.headers.get("X-Actor", "reviewer"), correlation_id=request.state.correlation_id,
        )
        return {**record, "fields": fields}

    @app.post("/api/template-registrations/{registration_id}/approve", tags=["compatibility"])
    async def compatibility_approve_registration(registration_id: str, request: Request) -> dict[str, Any]:
        return await approve_registration_common(registration_id, request)

    def create_document_record(
        *, content: bytes, filename: str, template_id: str | None, correlation_id: str
    ) -> dict[str, Any]:
        templates = [item for item in store.list("template") if item.get("status") == "active" and not item.get("deleted_at")]
        if template_id is None:
            if len(templates) != 1:
                raise HTTPException(status_code=409, detail="template match is ambiguous; select an approved template")
            template = templates[0]
            template_id = template["id"]
            confirmed = False
        else:
            template = _record_or_404(store, "template", template_id)
            confirmed = True
        if template.get("status") != "active":
            raise HTTPException(status_code=409, detail="template is not approved")
        document_id = _id("DOC")
        path = workflows.save_upload(document_id, filename, content)
        now = iso_now()
        record = {
            "id": document_id,
            "file_name": filename,
            "template_id": template_id,
            "template_version": template["version_id"],
            "form_type_id": template["form_type_id"],
            "source_path": str(path),
            "status": "uploaded",
            "sync_status": "not_synced",
            "review_status": "pending",
            "processed": None,
            "pages": 1,
            "progress": {"stage": "queued", "percent": 0},
            "template_match": {"template_id": template_id, "version": template["version"], "score": 1.0 if confirmed else 0.5, "confirmed": confirmed},
            "extraction_attempts": [],
            "human_corrections": [],
            "downstream_ids": {},
            "failure": None,
            "correlation_id": correlation_id,
            "created_at": now,
            "updated_at": now,
        }
        store.put("document", document_id, record, create_only=True)
        store.add_audit(
            action="uploaded completed form", target_type="document", target_id=document_id,
            correlation_id=correlation_id, template_version=template["version_id"], after={"file_name": filename},
        )
        return record

    @app.post("/api/v1/document-jobs", status_code=202, tags=["documents"])
    async def create_document_jobs(
        request: Request,
        background_tasks: BackgroundTasks,
        files: list[UploadFile] = File(...),
        template_id: str | None = Query(None),
    ) -> dict[str, Any]:
        items = []
        for upload in files:
            content = await _read_upload(upload, configured)
            item = create_document_record(
                content=content, filename=upload.filename or "document", template_id=template_id,
                correlation_id=request.state.correlation_id,
            )
            background_tasks.add_task(workflows.run_document, item["id"])
            items.append({"id": item["id"], "job_id": item["id"], "status": item["status"]})
        return {"items": items}

    @app.post("/api/v1/documents/process", status_code=202, tags=["documents"])
    async def legacy_process_document(
        request: Request,
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        template_id: str = Form(...),
    ) -> dict[str, Any]:
        content = await _read_upload(file, configured)
        item = create_document_record(
            content=content, filename=file.filename or "document", template_id=template_id,
            correlation_id=request.state.correlation_id,
        )
        background_tasks.add_task(workflows.run_document, item["id"])
        return {"job_id": item["id"], "status": item["status"]}

    @app.get("/api/v1/documents/{document_id}", tags=["documents"])
    @app.get("/api/v1/documents/jobs/{document_id}", tags=["documents"])
    async def canonical_get_document(document_id: str) -> dict[str, Any]:
        return _record_or_404(store, "document", document_id)

    @app.post("/api/v1/documents/{document_id}/template-match", tags=["documents"])
    async def override_template_match(document_id: str, request: Request, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        document = _record_or_404(store, "document", document_id)
        template_id = payload.get("template_id")
        template = _record_or_404(store, "template", str(template_id))
        before = document.get("template_match")
        document["template_id"] = template["id"]
        document["template_version"] = template["version_id"]
        document["template_match"] = {"template_id": template["id"], "version": template["version"], "score": 1.0, "confirmed": True, "reason": payload.get("reason")}
        document["updated_at"] = iso_now()
        store.put("document", document_id, document)
        store.add_audit(
            action="overrode template match", target_type="document", target_id=document_id,
            actor=request.headers.get("X-Actor", "reviewer"), before=before, after=document["template_match"],
            correlation_id=request.state.correlation_id, template_version=template["version_id"],
        )
        return document

    @app.post("/api/v1/documents/{document_id}/reprocess", status_code=202, tags=["documents"])
    async def reprocess_document(document_id: str, background_tasks: BackgroundTasks) -> dict[str, Any]:
        document = _record_or_404(store, "document", document_id)
        document["status"] = "uploaded"
        document["progress"] = {"stage": "queued", "percent": 0}
        store.put("document", document_id, document)
        background_tasks.add_task(workflows.run_document, document_id)
        return {"id": document_id, "status": "uploaded", "attempt": len(document["extraction_attempts"]) + 1}

    def apply_review(document: dict[str, Any], fields_payload: Any, actor: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        if isinstance(fields_payload, dict):
            corrections = [{"field_id": key, "corrected_value": value} for key, value in fields_payload.items()]
        elif isinstance(fields_payload, list):
            corrections = fields_payload
        else:
            raise HTTPException(status_code=422, detail="fields must be an object or list")
        processed_fields = (document.get("processed") or {}).setdefault("fields", {})
        immutable: list[dict[str, Any]] = []
        for correction in corrections:
            field_id = str(correction.get("field_id") or correction.get("key") or "")
            if not field_id:
                raise HTTPException(status_code=422, detail="correction field_id is required")
            field = processed_fields.setdefault(
                field_id,
                {"raw_value": "", "value": "", "normalized_value": "", "confidence": None, "source_region": None, "errors": [], "warnings": []},
            )
            original = field.get("value", field.get("raw_value", ""))
            corrected = correction.get("corrected_value", correction.get("value", ""))
            event = {
                "id": _id("COR"),
                "field_id": field_id,
                "original_value": original,
                "original_ocr_value": field.get("raw_value", original),
                "corrected_value": corrected,
                "source_region_id": correction.get("source_region_id", field.get("source_region")),
                "template_version": document["template_version"],
                "ocr_confidence": field.get("ocr_confidence", field.get("confidence")),
                "reviewer": actor,
                "reason": correction.get("reason"),
                "created_at": iso_now(),
            }
            immutable.append(event)
            field["value"] = str(corrected)
            field["normalized_value"] = str(corrected)
            field["source"] = "human_correction"
            field["review_status"] = "corrected"
            field["requires_review"] = False
            field["errors"] = []
        document["human_corrections"].extend(immutable)
        document["status"] = "needs_review"
        document["review_status"] = "in_review"
        document["updated_at"] = iso_now()
        return document, immutable

    @app.put("/api/v1/documents/{document_id}/review", tags=["documents"])
    async def review_document(document_id: str, request: Request, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        document = _record_or_404(store, "document", document_id)
        actor = request.headers.get("X-Actor", payload.get("reviewer", "reviewer"))
        document, corrections = apply_review(document, payload.get("fields"), actor)
        store.put("document", document_id, document)
        for correction in corrections:
            store.add_audit(
                action="corrected extracted field", target_type="document", target_id=document_id,
                actor=actor, before={"field_id": correction["field_id"], "value": correction["original_value"]},
                after={"field_id": correction["field_id"], "value": correction["corrected_value"]},
                correlation_id=request.state.correlation_id, template_version=document["template_version"],
                extraction_attempt=len(document["extraction_attempts"]),
            )
        return document

    def blocking_document_errors(document: dict[str, Any]) -> list[str]:
        errors = []
        if not document.get("template_match", {}).get("confirmed"):
            errors.append("low-confidence template match must be confirmed")
        for field_id, field in ((document.get("processed") or {}).get("fields") or {}).items():
            for message in field.get("errors", []):
                errors.append(f"{field_id}: {message}")
        return errors

    @app.post("/api/v1/documents/{document_id}/approve", tags=["documents"])
    async def approve_document(document_id: str, request: Request) -> dict[str, Any]:
        document = _record_or_404(store, "document", document_id)
        errors = blocking_document_errors(document)
        if errors:
            raise HTTPException(status_code=409, detail={"code": "APPROVAL_BLOCKED", "errors": errors})
        document["status"] = "ready_to_sync"
        document["review_status"] = "approved"
        document["approved_at"] = iso_now()
        document["updated_at"] = iso_now()
        store.put("document", document_id, document)
        store.add_audit(
            action="approved document", target_type="document", target_id=document_id,
            actor=request.headers.get("X-Actor", "reviewer"), correlation_id=request.state.correlation_id,
            template_version=document["template_version"], extraction_attempt=len(document["extraction_attempts"]),
        )
        return document

    @app.post("/api/v1/documents/{document_id}/sync", tags=["documents"])
    async def sync_document(document_id: str, request: Request) -> dict[str, Any]:
        document = _record_or_404(store, "document", document_id)
        if document.get("review_status") != "approved":
            raise HTTPException(status_code=409, detail="document must be approved before synchronization")
        document["status"] = "synced"
        document["sync_status"] = "synced"
        document["synced_at"] = iso_now()
        document["updated_at"] = iso_now()
        store.put("document", document_id, document)
        store.add_audit(
            action="synchronized document", target_type="document", target_id=document_id,
            actor=request.headers.get("X-Actor", "reviewer"), correlation_id=request.state.correlation_id,
            template_version=document["template_version"],
        )
        return document

    @app.get("/api/documents", tags=["compatibility"])
    async def list_documents() -> list[dict[str, Any]]:
        return [item for item in store.list("document") if not item.get("deleted_at")]

    @app.get("/api/documents/{document_id}", tags=["compatibility"])
    async def compatibility_get_document(document_id: str) -> dict[str, Any]:
        return _record_or_404(store, "document", document_id)

    @app.post("/api/documents", status_code=202, tags=["compatibility"])
    async def compatibility_upload_documents(
        request: Request,
        background_tasks: BackgroundTasks,
        template_id: str = Query(...),
        process_immediately: bool = Query(True),
        files: list[UploadFile] = File(...),
    ) -> dict[str, Any]:
        items = []
        for upload in files:
            content = await _read_upload(upload, configured)
            item = create_document_record(
                content=content, filename=upload.filename or "document", template_id=template_id,
                correlation_id=request.state.correlation_id,
            )
            if process_immediately:
                background_tasks.add_task(workflows.run_document, item["id"])
            items.append(item)
        return {"items": items}

    @app.patch("/api/documents/{document_id}/fields", tags=["compatibility"])
    async def compatibility_review_document(document_id: str, request: Request, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        document = _record_or_404(store, "document", document_id)
        document, corrections = apply_review(document, payload.get("fields"), request.headers.get("X-Actor", "reviewer"))
        store.put("document", document_id, document)
        for correction in corrections:
            store.add_audit(
                action="saved human correction", target_type="document", target_id=document_id,
                actor=correction["reviewer"], before={"field_id": correction["field_id"], "value": correction["original_value"]},
                after={"field_id": correction["field_id"], "value": correction["corrected_value"]},
                correlation_id=request.state.correlation_id, template_version=document["template_version"],
            )
        return document

    @app.post("/api/documents/{document_id}/status", tags=["compatibility"])
    async def compatibility_document_status(document_id: str, request: Request, payload: dict[str, str] = Body(...)) -> dict[str, Any]:
        status_value = payload.get("status")
        if status_value not in {"uploaded", "processing", "needs_review", "ready_to_sync", "synced", "failed"}:
            raise HTTPException(status_code=400, detail="invalid status")
        document = _record_or_404(store, "document", document_id)
        if status_value in {"ready_to_sync", "synced"}:
            errors = blocking_document_errors(document)
            if errors:
                raise HTTPException(status_code=409, detail={"code": "APPROVAL_BLOCKED", "errors": errors})
        document["status"] = status_value
        if status_value == "synced":
            document["sync_status"] = "synced"
            document["review_status"] = "approved"
        document["updated_at"] = iso_now()
        store.put("document", document_id, document)
        store.add_audit(
            action=f"changed status to {status_value}", target_type="document", target_id=document_id,
            actor=request.headers.get("X-Actor", "reviewer"), correlation_id=request.state.correlation_id,
        )
        return document

    @app.delete("/api/documents/{document_id}", tags=["compatibility"])
    async def compatibility_delete_document(document_id: str, request: Request) -> dict[str, str]:
        document = _record_or_404(store, "document", document_id)
        document["deleted_at"] = iso_now()
        document["updated_at"] = iso_now()
        store.put("document", document_id, document)
        store.add_audit(
            action="archived document", target_type="document", target_id=document_id,
            actor=request.headers.get("X-Actor", "reviewer"), correlation_id=request.state.correlation_id,
        )
        return {"status": "deleted", "id": document_id}

    @app.get("/api/audit-events", tags=["compatibility"])
    async def audit_events() -> list[dict[str, Any]]:
        return store.list("audit")

    def export_rows() -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for document in store.list("document"):
            if document.get("deleted_at"):
                continue
            fields = (document.get("processed") or {}).get("fields", {})
            row = {
                "document_id": document["id"],
                "file_name": document["file_name"],
                "form_type": document["form_type_id"],
                "status": document["status"],
                "sync_status": document["sync_status"],
            }
            row.update({key: value.get("value", "") for key, value in fields.items()})
            rows.append(row)
        return rows

    @app.get("/api/export/json", tags=["compatibility"])
    async def export_json() -> dict[str, Any]:
        return {
            "exported_at": iso_now(),
            "templates": [item for item in store.list("template") if not item.get("deleted_at")],
            "registrations": [item for item in store.list("registration") if not item.get("deleted_at")],
            "documents": [item for item in store.list("document") if not item.get("deleted_at")],
        }

    @app.get("/api/export/csv", tags=["compatibility"])
    async def export_csv() -> Response:
        rows = export_rows()
        headers = sorted({key for row in rows for key in row}) or ["document_id"]
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
        return Response(
            content="\ufeff" + output.getvalue(), media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": "attachment; filename=insurance-records.csv"},
        )

    @app.get("/api/export/excel", tags=["compatibility"])
    async def export_excel() -> Response:
        rows = export_rows()
        headers = sorted({key for row in rows for key in row}) or ["document_id"]
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "Insurance Records"
        sheet.append(headers)
        for row in rows:
            sheet.append([row.get(header, "") for header in headers])
        output = io.BytesIO()
        workbook.save(output)
        return Response(
            content=output.getvalue(), media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": "attachment; filename=insurance-records.xlsx"},
        )

    @app.get("/api/v1/documents/{document_id}/export/{export_format}", tags=["documents"])
    async def proxy_document_export(document_id: str, export_format: str, request: Request):
        document = _record_or_404(store, "document", document_id)
        upstream_job = document.get("downstream_ids", {}).get("document_job_id")
        if not upstream_job:
            raise HTTPException(status_code=409, detail="document processing is not complete")
        if export_format not in {"json", "csv", "excel", "xlsx"}:
            raise HTTPException(status_code=400, detail="unsupported export format")
        response = await client.request(
            "document-processing-layer", "GET",
            f"{configured.document_processing_url}/api/v1/documents/jobs/{upstream_job}/export/{export_format}",
            correlation_id=request.state.correlation_id,
        )
        return Response(
            content=response.content,
            media_type=response.headers.get("content-type", "application/octet-stream"),
            headers={"Content-Disposition": response.headers.get("content-disposition", f"attachment; filename={document_id}.{export_format}")},
        )

    return app


app = create_app()
