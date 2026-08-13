from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import uuid
from pathlib import Path
from typing import Any

from adapters.contracts import (
    AdapterError,
    ImageIdentity,
    VLM_FIELD_TYPES,
    build_vlm_contracts,
    semantic_draft_to_template,
    xyxy_to_normalized_xywh,
)

from .config import Settings
from .downstream import DownstreamClient, DownstreamError
from .store import RecordStore, iso_now


logger = logging.getLogger("orchestrator.workflows")


LAYOUT_FALLBACK_MODES = {
    "TEXT_INPUT_BOX": "printed",
    "MULTILINE_BOX": "printed",
    "INPUT_LINE": "handwriting",
    "CHECKBOX": "checkbox",
    "RADIO_BUTTON": "checkbox",
    "SIGNATURE_AREA": "signature",
    "TABLE": "table",
    "TABLE_CELL": "table",
}
PREPROCESSING_MODES = {
    "auto": "auto",
    "force": "standard",
    "none": "none",
}

class WorkflowService:
    def __init__(self, settings: Settings, store: RecordStore, client: DownstreamClient):
        self.settings = settings
        self.store = store
        self.client = client

    def save_upload(self, record_id: str, filename: str, content: bytes) -> Path:
        suffix = Path(filename).suffix.lower() or ".bin"
        target_dir = self.settings.storage_root / record_id
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"source{suffix}"
        target.write_bytes(content)
        return target

    async def run_template_registration(self, registration_id: str) -> None:
        record = self.store.require("registration", registration_id)
        correlation_id = record["correlation_id"]
        try:
            self._template_progress(
                record,
                "upload_validation",
                8,
                status="validating",
                message="Creating the visual-service registration document.",
            )
            source_path = Path(record["source_path"])
            source_bytes = source_path.read_bytes()
            media_type = mimetypes.guess_type(record["file_name"])[0] or "application/octet-stream"
            response = await self.client.request(
                "visual-field-detection",
                "POST",
                f"{self.settings.visual_field_url}/v1/documents",
                correlation_id=correlation_id,
                files={"file": (record["file_name"], source_bytes, media_type)},
            )
            visual_document = response.json()
            record["downstream_ids"]["visual_document_id"] = visual_document["document_id"]

            self._template_progress(
                record,
                "preprocessing",
                18,
                status="preprocessing",
                message="Creating and assessing the canonical template page.",
            )
            visual_id = visual_document["document_id"]
            canonical_page = await self._preprocess_registration(record, visual_id, correlation_id)
            if canonical_page is None:
                return
            page_bytes, identity = canonical_page

            self._template_progress(
                record,
                "layout_and_ocr",
                38,
                status="extracting",
                message="Running layout detection and printed-label OCR in parallel.",
            )
            record["layout_status"] = "running"
            record["ocr_status"] = "running"
            record["updated_at"] = iso_now()
            self.store.put("registration", registration_id, record)

            layout_task = asyncio.create_task(
                self._run_layout_branch(record, visual_id, correlation_id)
            )
            ocr_task = asyncio.create_task(
                self._run_ocr_branch(record, page_bytes, correlation_id)
            )
            branch_results = await asyncio.gather(
                layout_task,
                ocr_task,
                return_exceptions=True,
            )
            for branch_result in branch_results:
                if isinstance(branch_result, Exception):
                    raise branch_result
            layout_response, ocr_response = branch_results
            if len(layout_response.get("pages", [])) != 1:
                raise AdapterError("Template registration currently supports exactly one page")

            self._template_progress(
                record,
                "contract_validation",
                60,
                status="contract_validation",
                message="Validating canonical identity, geometry, and downstream contracts.",
            )
            ocr_contract, layout_contract, adapter_warnings = build_vlm_contracts(
                ocr_response, layout_response, identity
            )
            record["adapter_warnings"] = adapter_warnings
            record["layout_contract"] = layout_contract

            self._template_progress(
                record,
                "semantic_mapping",
                70,
                status="vlm_queued",
                message="Submitting validated contracts to Insurance-VLM.",
            )
            vlm_headers = {"X-API-Key": self.settings.vlm_api_key} if self.settings.vlm_api_key else {}
            vlm_response = await self.client.request(
                "insurance-vlm",
                "POST",
                f"{self.settings.vlm_url}/api/v1/registrations",
                correlation_id=correlation_id,
                headers=vlm_headers,
                files={
                    "image": ("page_001.png", page_bytes, "image/png"),
                    "ocr_json": ("ocr-output.json", json.dumps(ocr_contract).encode(), "application/json"),
                    "layout_json": ("layout-output.json", json.dumps(layout_contract).encode(), "application/json"),
                },
            )
            vlm_job = vlm_response.json()
            record["downstream_ids"]["vlm_job_id"] = vlm_job["job_id"]
            self.store.put("registration", registration_id, record)

            self._template_progress(
                record,
                "vlm_poll",
                82,
                status="vlm_running",
                message="Insurance-VLM semantic mapping is running.",
            )
            result = await self._poll_vlm(vlm_job["job_id"], correlation_id, vlm_headers)
            record["vlm_result"] = result
            record["draft"] = self._build_editable_draft(record, result, layout_contract)
            record["draft_revision"] = 1
            record["status"] = "needs_approval"
            record["progress"] = {"stage": "human_review", "percent": 100}
            record["updated_at"] = iso_now()
            self.store.put("registration", registration_id, record)
            self.store.add_audit(
                action="completed semantic registration draft",
                target_type="template_registration",
                target_id=registration_id,
                correlation_id=correlation_id,
                after={"status": record["status"], "draft_revision": 1},
            )
        except (AdapterError, DownstreamError, KeyError, ValueError) as exc:
            logger.warning("template registration failed", extra={"registration_id": registration_id, "correlation_id": correlation_id})
            record["status"] = "failed"
            record["progress"] = {"stage": "failed", "percent": record.get("progress", {}).get("percent", 0)}
            record["failure"] = self._safe_failure(exc)
            record["updated_at"] = iso_now()
            self.store.put("registration", registration_id, record)
            self.store.add_audit(
                action="template registration failed",
                target_type="template_registration",
                target_id=registration_id,
                correlation_id=correlation_id,
                after=record["failure"],
            )

    async def _preprocess_registration(
        self,
        record: dict[str, Any],
        visual_id: str,
        correlation_id: str,
    ) -> tuple[bytes, ImageIdentity] | None:
        requested_policy = str(record.get("preprocessing", {}).get("requested_policy", "auto"))
        correction_mode = PREPROCESSING_MODES.get(requested_policy)
        if correction_mode is None:
            raise ValueError(f"Unsupported preprocessing policy: {requested_policy}")

        preprocess_response = await self.client.request(
            "visual-field-detection",
            "POST",
            f"{self.settings.visual_field_url}/v1/documents/{visual_id}/preprocess",
            correlation_id=correlation_id,
            json={
                "correction_mode": correction_mode,
                "capture_profile": "template",
                "deskew": True,
                "normalize_illumination": True,
                "sharpen": True,
            },
        )
        preprocess_result = preprocess_response.json()
        summary = preprocess_result.get("summary") or {}
        if summary.get("page_count") != 1:
            raise AdapterError("Template registration currently supports exactly one page")

        artifact_path = str(preprocess_result.get("artifact") or "")
        if not artifact_path or artifact_path.startswith("/") or ".." in Path(artifact_path).parts:
            raise AdapterError("Visual-field preprocessing did not return a valid manifest artifact")
        preprocess_manifest = (
            await self.client.request(
                "visual-field-detection",
                "GET",
                f"{self.settings.visual_field_url}/v1/documents/{visual_id}/artifacts/{artifact_path}",
                correlation_id=correlation_id,
            )
        ).json()

        preprocessing = self._build_preprocessing_result(
            requested_policy,
            correction_mode,
            preprocess_manifest,
        )
        record["preprocessing"] = preprocessing
        record["updated_at"] = iso_now()
        self.store.put("registration", record["id"], record)

        if preprocessing["retake_required"]:
            record["status"] = "needs_resubmission"
            record["progress"] = {
                "stage": "capture_quality",
                "percent": 25,
                "message": "The template image did not pass capture quality checks. Please upload a new image.",
            }
            record["updated_at"] = iso_now()
            self.store.put("registration", record["id"], record)
            self.store.add_audit(
                action="template registration requires resubmission",
                target_type="template_registration",
                target_id=record["id"],
                correlation_id=correlation_id,
                after={
                    "status": record["status"],
                    "decision": preprocessing["decision"],
                    "reasons": preprocessing["reasons"],
                },
            )
            return None

        manifest_page = preprocess_manifest["pages"][0]
        image_path = str(manifest_page.get("image_path") or "")
        if not image_path or image_path.startswith("/") or ".." in Path(image_path).parts:
            raise AdapterError("Preprocessing manifest did not identify a valid canonical page artifact")
        page_response = await self.client.request(
            "visual-field-detection",
            "GET",
            f"{self.settings.visual_field_url}/v1/documents/{visual_id}/artifacts/{image_path}",
            correlation_id=correlation_id,
        )
        page_bytes = page_response.content
        if not page_bytes:
            raise AdapterError("Canonical page artifact is empty")

        page_id = str(manifest_page.get("page_id") or "")
        page_number = manifest_page.get("page_number")
        if not page_id or page_number != 1:
            raise AdapterError("Preprocessing manifest has an invalid canonical page identity")
        identity = ImageIdentity.from_bytes(
            page_bytes,
            document_id=visual_id,
            page_id=page_id,
            page_number=page_number,
        )
        if manifest_page.get("sha256") != identity.sha256:
            raise AdapterError("Canonical page SHA-256 does not match the preprocessing manifest")
        if manifest_page.get("width") != identity.width or manifest_page.get("height") != identity.height:
            raise AdapterError("Canonical page dimensions do not match the preprocessing manifest")

        canonical_path = self.settings.storage_root / record["id"] / "page_001.png"
        canonical_path.write_bytes(page_bytes)
        record["image_identity"] = {
            "sha256": identity.sha256,
            "width": identity.width,
            "height": identity.height,
            "document_id": identity.document_id,
            "page_id": identity.page_id,
            "page_number": identity.page_number,
        }
        record["page_images"] = [f"/api/v1/template-registrations/{record['id']}/pages/1"]
        record["canonical_artifact"] = {
            "upstream_path": image_path,
            "retrieved_at": iso_now(),
        }
        record["updated_at"] = iso_now()
        self.store.put("registration", record["id"], record)
        self.store.add_audit(
            action="established canonical template page",
            target_type="template_registration",
            target_id=record["id"],
            correlation_id=correlation_id,
            after=record["image_identity"],
        )
        return page_bytes, identity

    @staticmethod
    def _build_preprocessing_result(
        requested_policy: str,
        correction_mode: str,
        manifest: dict[str, Any],
    ) -> dict[str, Any]:
        pages = manifest.get("pages") or []
        if len(pages) != 1:
            raise AdapterError("Template registration currently supports exactly one page")

        decision = manifest.get("decision")
        if decision not in {"canonicalize_only", "correct", "reject_and_retake"}:
            raise AdapterError("Preprocessing manifest did not provide a valid authoritative decision")
        capture_ready = manifest.get("capture_ready")
        retake_required = manifest.get("retake_required")
        quality_pass = manifest.get("quality_pass")
        if any(type(value) is not bool for value in (capture_ready, retake_required, quality_pass)):
            raise AdapterError("Preprocessing manifest decision flags must be boolean")
        if retake_required == capture_ready:
            raise AdapterError("Preprocessing manifest has inconsistent capture decision flags")
        if (decision == "reject_and_retake") != retake_required:
            raise AdapterError("Preprocessing manifest decision conflicts with its retake flag")
        if capture_ready and not quality_pass:
            raise AdapterError("Preprocessing manifest cannot accept a page that failed quality checks")

        authoritative_lists: dict[str, list[str]] = {}
        for field in ("reasons", "advisories", "instructions", "operations"):
            value = manifest.get(field)
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                raise AdapterError(f"Preprocessing manifest {field} must be a list of strings")
            authoritative_lists[field] = list(dict.fromkeys(value))

        return {
            "requested_policy": requested_policy,
            "correction_mode": correction_mode,
            "decision": decision,
            "capture_ready": capture_ready,
            "retake_required": retake_required,
            "reasons": authoritative_lists["reasons"],
            "advisories": authoritative_lists["advisories"],
            "instructions": authoritative_lists["instructions"],
            "operations": authoritative_lists["operations"],
            "quality_pass": quality_pass,
            "capture_profile": manifest.get("capture_profile"),
            "page_count": int(manifest.get("page_count", len(pages))),
            "pages": [
                {
                    "page_id": page.get("page_id"),
                    "page_number": page.get("page_number"),
                    "image_path": page.get("image_path"),
                    "width": page.get("width"),
                    "height": page.get("height"),
                    "sha256": page.get("sha256"),
                    "source_to_page_transform": page.get("source_to_page_transform"),
                    "quality": page.get("quality"),
                }
                for page in pages
            ],
        }

    async def _run_layout_branch(
        self,
        record: dict[str, Any],
        visual_id: str,
        correlation_id: str,
    ) -> dict[str, Any]:
        try:
            await self.client.request(
                "visual-field-detection",
                "POST",
                f"{self.settings.visual_field_url}/v1/documents/{visual_id}/extract",
                correlation_id=correlation_id,
                json={"detect_table_cells": True},
            )
            result = (
                await self.client.request(
                    "visual-field-detection",
                    "GET",
                    f"{self.settings.visual_field_url}/v1/documents/{visual_id}/result",
                    correlation_id=correlation_id,
                )
            ).json()
        except Exception:
            self._set_branch_status(record, "layout", "failed")
            raise
        self._set_branch_status(record, "layout", "complete")
        return result

    async def _run_ocr_branch(
        self,
        record: dict[str, Any],
        page_bytes: bytes,
        correlation_id: str,
    ) -> dict[str, Any]:
        try:
            result = (
                await self.client.request(
                    "ocr-fastapi-service",
                    "POST",
                    f"{self.settings.ocr_url}/v1/ocr/process",
                    correlation_id=correlation_id,
                    params={"preprocess_mode": "minimal", "language": "eng+mya"},
                    files={"file": ("page_001.png", page_bytes, "image/png")},
                )
            ).json()
        except Exception:
            self._set_branch_status(record, "ocr", "failed")
            raise
        self._set_branch_status(record, "ocr", "complete")
        return result

    def _set_branch_status(
        self,
        record: dict[str, Any],
        branch: str,
        status: str,
    ) -> None:
        record[f"{branch}_status"] = status
        record["updated_at"] = iso_now()
        self.store.put("registration", record["id"], record)

    async def _poll_vlm(self, job_id: str, correlation_id: str, headers: dict[str, str]) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.settings.poll_timeout_seconds
        while loop.time() < deadline:
            status = (
                await self.client.request(
                    "insurance-vlm",
                    "GET",
                    f"{self.settings.vlm_url}/api/v1/registrations/{job_id}",
                    correlation_id=correlation_id,
                    headers=headers,
                )
            ).json()
            if status.get("status") == "COMPLETED":
                return (
                    await self.client.request(
                        "insurance-vlm",
                        "GET",
                        f"{self.settings.vlm_url}/api/v1/registrations/{job_id}/result",
                        correlation_id=correlation_id,
                        headers=headers,
                    )
                ).json()
            if status.get("status") == "FAILED":
                raise DownstreamError("insurance-vlm", "semantic registration failed", 502, detail=status.get("error"))
            await asyncio.sleep(self.settings.poll_interval_seconds)
        raise DownstreamError("insurance-vlm", "semantic registration polling timed out", 504)

    def _build_editable_draft(
        self, record: dict[str, Any], result: dict[str, Any], layout_contract: dict[str, Any]
    ) -> dict[str, Any]:
        semantic = result.get("semantic_output") or {}
        pages = semantic.get("pages") or []
        semantic_page = pages[0] if pages else {}
        labels = {item.get("label_id"): item for item in semantic_page.get("semantic_labels", [])}
        fields_by_region: dict[str, dict[str, Any]] = {}
        for field in semantic_page.get("fields", []):
            for region_id in field.get("region_ids", []):
                fields_by_region[region_id] = field
        identity = record["image_identity"]
        draft_regions: list[dict[str, Any]] = []
        for index, layout_region in enumerate(layout_contract["pages"][0]["regions"], 1):
            if layout_region["region_type"] == "TABLE_CELL":
                continue
            semantic_field = fields_by_region.get(layout_region["region_id"])
            label = labels.get((semantic_field or {}).get("label_id"), {})
            field_type = str((semantic_field or {}).get("field_type", ""))
            extraction_mode = VLM_FIELD_TYPES.get(field_type) or LAYOUT_FALLBACK_MODES.get(layout_region["region_type"])
            review_flags = []
            if semantic_field is None:
                review_flags.append("VLM did not assign this layout region; human confirmation is required")
            if extraction_mode is None:
                review_flags.append(f"Unsupported or ambiguous field type: {field_type or layout_region['region_type']}")
            key = (semantic_field or {}).get("key") or f"field_{index:03d}"
            draft_regions.append(
                {
                    "id": layout_region["region_id"],
                    "field_id": (semantic_field or {}).get("field_id") or key,
                    "page": 1,
                    "key": key,
                    "label": label.get("primary_text") or key.replace("_", " ").title(),
                    "data_type": field_type or "text",
                    "language": label.get("primary_language") or "unknown",
                    "extraction_mode": extraction_mode,
                    "required": bool((semantic_field or {}).get("required", False)),
                    "confidence": float((semantic_field or {}).get("confidence", layout_region.get("confidence", 0))),
                    "bbox": xyxy_to_normalized_xywh(layout_region["bbox_px"], identity["width"], identity["height"]),
                    "source_region_ids": list((semantic_field or {}).get("region_ids", [layout_region["region_id"]])),
                    "parent_region_id": layout_region.get("parent_region_id"),
                    "review_flags": review_flags,
                    "validation": (semantic_field or {}).get("validation"),
                }
            )
        return {
            "revision": 1,
            "regions": draft_regions,
            "warnings": semantic.get("warnings", []) + record.get("adapter_warnings", []),
            "quality_summary": result.get("quality_summary", {}),
        }

    def _template_progress(
        self,
        record: dict[str, Any],
        stage: str,
        percent: int,
        *,
        status: str = "analyzing",
        message: str | None = None,
    ) -> None:
        record["status"] = status
        record["progress"] = {"stage": stage, "percent": percent}
        if message:
            record["progress"]["message"] = message
        record["updated_at"] = iso_now()
        self.store.put("registration", record["id"], record)

    async def approve_template(self, registration_id: str, actor: str, correlation_id: str) -> dict[str, Any]:
        record = self.store.require("registration", registration_id)
        if record.get("status") == "registered":
            template = self.store.get("template", record["template_id"])
            if template:
                return {"registration": record, "template": template}
        if record.get("status") != "needs_approval":
            raise ValueError("registration is not ready for approval")
        revision = int(record.get("approved_version_number", 0)) + 1
        template_id = record.get("template_id") or f"TPL-{uuid.uuid4().hex[:10].upper()}"
        pinned_id = f"{template_id.lower().replace('-', '_')}_v{revision}"
        identity = record["image_identity"]
        definition, review_flags = semantic_draft_to_template(
            template_id=pinned_id,
            name=record.get("name") or f"{record['form_type_id'].title()} Claim Template",
            width=identity["width"],
            height=identity["height"],
            regions=record["draft"]["regions"],
        )
        if review_flags:
            raise AdapterError("; ".join(review_flags))
        response = await self.client.request(
            "document-processing-layer",
            "POST",
            f"{self.settings.document_processing_url}/api/v1/templates/register",
            correlation_id=correlation_id,
            json=definition,
        )
        registered_definition = response.json()
        reference_path = self.settings.storage_root / registration_id / "page_001.png"
        if not reference_path.is_file():
            raise AdapterError("Approved template is missing its canonical reference page")
        await self.client.request(
            "document-processing-layer",
            "POST",
            f"{self.settings.document_processing_url}/api/v1/templates/{pinned_id}/reference",
            correlation_id=correlation_id,
            files={"file": ("page_001.png", reference_path.read_bytes(), "image/png")},
        )
        now = iso_now()
        version_id = f"{template_id}:v{revision}"
        version = {
            "id": version_id,
            "template_id": template_id,
            "version": str(revision),
            "definition": registered_definition,
            "draft_snapshot": record["draft"],
            "reference_image_path": str(reference_path),
            "approved_at": now,
            "approved_by": actor,
            "immutable": True,
        }
        self.store.put("template_version", version_id, version, create_only=True)
        template = {
            "id": template_id,
            "name": record.get("name") or f"{record['form_type_id'].title()} Claim Template",
            "form_type_id": record["form_type_id"],
            "version": str(revision),
            "version_id": version_id,
            "downstream_template_id": pinned_id,
            "status": "active",
            "confidence_score": record.get("vlm_result", {}).get("quality_summary", {}).get("actionable_coverage_ratio", 0),
            "fields": [item["key"] for item in record["draft"]["regions"]],
            "source_file": record["file_name"],
            "created_at": record["created_at"],
            "updated_at": now,
        }
        self.store.put("template", template_id, template)
        record["template_id"] = template_id
        record["approved_version_number"] = revision
        record["status"] = "registered"
        record["approved_at"] = now
        record["updated_at"] = now
        self.store.put("registration", registration_id, record)
        self.store.add_audit(
            action="approved template",
            target_type="template",
            target_id=template_id,
            actor=actor,
            correlation_id=correlation_id,
            template_version=version_id,
            after={"version": revision, "downstream_template_id": pinned_id},
        )
        return {"registration": record, "template": template}

    async def run_document(self, document_id: str) -> None:
        record = self.store.require("document", document_id)
        correlation_id = record["correlation_id"]
        try:
            template = self.store.require("template", record["template_id"])
            if template.get("status") != "active" or not template.get("version_id"):
                raise ValueError("template has not been approved and registered")
            template_version = self.store.require("template_version", template["version_id"])
            definition = template_version.get("definition")
            if not isinstance(definition, dict):
                raise ValueError("approved template version has no registered definition")
            # The current document-processing service keeps its registry in memory.
            # Re-registering the immutable approved definition is idempotent and
            # restores that registry after a downstream container restart.
            await self.client.request(
                "document-processing-layer",
                "POST",
                f"{self.settings.document_processing_url}/api/v1/templates/register",
                correlation_id=correlation_id,
                json=definition,
            )
            reference_image_path = template_version.get("reference_image_path")
            if not isinstance(reference_image_path, str) or not Path(reference_image_path).is_file():
                raise AdapterError("approved template version is missing its canonical reference image")
            reference_path = Path(reference_image_path)
            await self.client.request(
                "document-processing-layer",
                "POST",
                f"{self.settings.document_processing_url}/api/v1/templates/{template['downstream_template_id']}/reference",
                correlation_id=correlation_id,
                files={"file": ("page_001.png", reference_path.read_bytes(), "image/png")},
            )
            record["status"] = "processing"
            record["progress"] = {"stage": "document_processing", "percent": 35}
            record["updated_at"] = iso_now()
            self.store.put("document", document_id, record)
            source_path = Path(record["source_path"])
            response = await self.client.request(
                "document-processing-layer",
                "POST",
                f"{self.settings.document_processing_url}/api/v1/documents/process",
                correlation_id=correlation_id,
                files={"file": (record["file_name"], source_path.read_bytes(), mimetypes.guess_type(record["file_name"])[0] or "application/octet-stream")},
                data={"template_id": template["downstream_template_id"]},
            )
            upstream = response.json()
            record["downstream_ids"]["document_job_id"] = upstream["job_id"]
            record["processed"] = self._adapt_document_result(upstream)
            record["extraction_attempts"].append(
                {"attempt": len(record["extraction_attempts"]) + 1, "created_at": iso_now(), "upstream": upstream}
            )
            record["status"] = "needs_review" if upstream.get("needs_human_review", False) else "ready_to_sync"
            record["review_status"] = "pending"
            record["progress"] = {"stage": "human_review", "percent": 100}
            record["export_urls"] = {
                fmt: f"/api/v1/documents/{document_id}/export/{fmt}" for fmt in ("json", "csv", "excel")
            }
            record["updated_at"] = iso_now()
            self.store.put("document", document_id, record)
            self.store.add_audit(
                action="processed completed document",
                target_type="document",
                target_id=document_id,
                correlation_id=correlation_id,
                template_version=template["version_id"],
                extraction_attempt=len(record["extraction_attempts"]),
                after={"status": record["status"], "upstream_job_id": upstream["job_id"]},
            )
        except (DownstreamError, KeyError, ValueError) as exc:
            record["status"] = "failed"
            record["failure"] = self._safe_failure(exc)
            record["progress"] = {"stage": "failed", "percent": record.get("progress", {}).get("percent", 0)}
            record["updated_at"] = iso_now()
            self.store.put("document", document_id, record)
            self.store.add_audit(
                action="document processing failed",
                target_type="document",
                target_id=document_id,
                correlation_id=correlation_id,
                after=record["failure"],
            )

    @staticmethod
    def _adapt_document_result(upstream: dict[str, Any]) -> dict[str, Any]:
        fields: dict[str, Any] = {}
        review_fields: list[str] = []
        for item in upstream.get("extracted_fields", []):
            key = item.get("field_id", f"field_{len(fields) + 1}")
            warnings = [item["validation_message"]] if item.get("validation_message") else []
            errors = [] if item.get("validation_passed", False) else warnings or ["Validation failed"]
            fields[key] = {
                "raw_value": item.get("raw_text", ""),
                "value": item.get("normalized_text", ""),
                "normalized_value": item.get("normalized_text", ""),
                "confidence": item.get("final_confidence", item.get("ocr_confidence", 0)),
                "ocr_confidence": item.get("ocr_confidence", 0),
                "source": "document-processing-layer",
                "source_region": item.get("crop_image_path"),
                "field_type": item.get("field_type", "printed_text"),
                "is_valid": bool(item.get("validation_passed", False)),
                "validation_status": "valid" if item.get("validation_passed", False) else "invalid",
                "review_status": "required" if item.get("human_review_flag", False) else "not_required",
                "requires_review": bool(item.get("human_review_flag", False)),
                "errors": errors,
                "warnings": warnings,
                "input_field": key,
                "label": item.get("label", key.replace("_", " ").title()),
            }
            if item.get("human_review_flag", False):
                review_fields.append(key)
        return {
            "fields": fields,
            "quality_check": upstream.get("quality_check"),
            "summary": {
                "overall_confidence": upstream.get("overall_confidence", 0),
                "review_fields": review_fields,
                "needs_human_review": upstream.get("needs_human_review", False),
            },
            "original_upstream_result": upstream,
        }

    @staticmethod
    def _safe_failure(exc: Exception) -> dict[str, Any]:
        if isinstance(exc, DownstreamError):
            return {
                "code": "DOWNSTREAM_ERROR",
                "service": exc.service,
                "message": exc.message,
                "upstream_status": exc.upstream_status,
                "details": exc.detail,
            }
        if isinstance(exc, AdapterError):
            return {"code": "CONTRACT_ERROR", "message": str(exc)}
        return {"code": "WORKFLOW_ERROR", "message": str(exc)}
