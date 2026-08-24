from __future__ import annotations

import asyncio
import io
import json
import logging
import mimetypes
import uuid
from pathlib import Path
from typing import Any

from PIL import Image

from adapters.contracts import (
    AdapterError,
    ImageIdentity,
    VLM_FIELD_TYPES,
    build_vlm_contracts,
    semantic_draft_to_template,
    validate_vlm_relationships,
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
            canonical_pages = await self._preprocess_registration(record, visual_id, correlation_id)
            if canonical_pages is None:
                return

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
            ocr_tasks = [
                asyncio.create_task(
                    self._run_ocr_branch(record, page_bytes, identity, correlation_id)
                )
                for page_bytes, identity in canonical_pages
            ]
            branch_results = await asyncio.gather(
                layout_task,
                *ocr_tasks,
                return_exceptions=True,
            )
            self._set_branch_status(
                record,
                "layout",
                "failed" if isinstance(branch_results[0], Exception) else "complete",
            )
            self._set_branch_status(
                record,
                "ocr",
                "failed" if any(isinstance(item, Exception) for item in branch_results[1:]) else "complete",
            )
            for branch_result in branch_results:
                if isinstance(branch_result, Exception):
                    raise branch_result
            layout_response = branch_results[0]
            ocr_responses = branch_results[1:]
            layout_pages = layout_response.get("pages", [])
            if len(layout_pages) != len(canonical_pages):
                raise AdapterError("Layout page count does not match the canonical template")

            self._template_progress(
                record,
                "contract_validation",
                60,
                status="contract_validation",
                message="Validating canonical identity, geometry, and downstream contracts.",
            )
            page_contracts: list[dict[str, Any]] = []
            adapter_warnings: list[str] = []
            for (page_bytes, identity), ocr_response in zip(canonical_pages, ocr_responses):
                matching_layout_pages = [
                    page
                    for page in layout_pages
                    if page.get("page_id") == identity.page_id
                    and page.get("page_number") == identity.page_number
                ]
                if len(matching_layout_pages) != 1:
                    raise AdapterError(
                        f"Layout did not return exactly one canonical {identity.page_id} page"
                    )
                page_layout_response = {
                    **layout_response,
                    "pages": matching_layout_pages,
                }
                ocr_contract, layout_contract, page_warnings = build_vlm_contracts(
                    ocr_response,
                    page_layout_response,
                    identity,
                )
                page_contracts.append({
                    "page_bytes": page_bytes,
                    "identity": identity,
                    "ocr_contract": ocr_contract,
                    "layout_contract": layout_contract,
                })
                adapter_warnings.extend(page_warnings)
            combined_layout_contract = {
                **page_contracts[0]["layout_contract"],
                "pages": [
                    context["layout_contract"]["pages"][0]
                    for context in page_contracts
                ],
            }
            record["adapter_warnings"] = list(dict.fromkeys(adapter_warnings))
            record["layout_contract"] = combined_layout_contract

            self._template_progress(
                record,
                "semantic_mapping",
                70,
                status="vlm_queued",
                message="Submitting validated contracts to Insurance-VLM.",
            )
            vlm_headers = {"X-API-Key": self.settings.vlm_api_key} if self.settings.vlm_api_key else {}
            vlm_results: list[dict[str, Any]] = []
            vlm_jobs: list[dict[str, Any]] = []
            for page_index, context in enumerate(page_contracts, 1):
                identity = context["identity"]
                vlm_response = await self.client.request(
                    "insurance-vlm",
                    "POST",
                    f"{self.settings.vlm_url}/api/v1/registrations",
                    correlation_id=correlation_id,
                    headers=vlm_headers,
                    files={
                        "image": (
                            f"{identity.page_id}.png",
                            context["page_bytes"],
                            "image/png",
                        ),
                        "ocr_json": (
                            f"{identity.page_id}-ocr.json",
                            json.dumps(context["ocr_contract"]).encode(),
                            "application/json",
                        ),
                        "layout_json": (
                            f"{identity.page_id}-layout.json",
                            json.dumps(context["layout_contract"]).encode(),
                            "application/json",
                        ),
                    },
                )
                vlm_job = vlm_response.json()
                vlm_jobs.append({
                    "page_id": identity.page_id,
                    "page_number": identity.page_number,
                    "job_id": vlm_job["job_id"],
                })
                record["downstream_ids"]["vlm_jobs"] = vlm_jobs
                record["downstream_ids"]["vlm_job_ids"] = [
                    item["job_id"] for item in vlm_jobs
                ]
                record["downstream_ids"]["vlm_job_id"] = vlm_jobs[0]["job_id"]
                self.store.put("registration", registration_id, record)
                self._template_progress(
                    record,
                    "vlm_poll",
                    75 + round(12 * page_index / len(page_contracts)),
                    status="vlm_running",
                    message=(
                        "Insurance-VLM semantic mapping is running "
                        f"for page {page_index} of {len(page_contracts)}."
                    ),
                )
                result = await self._poll_vlm(
                    vlm_job["job_id"], correlation_id, vlm_headers
                )
                vlm_results.append(result)
                context["result"] = result
            self._template_progress(
                record,
                "relationship_validation",
                92,
                status="relationship_validation",
                message="Validating VLM references, coverage, and authoritative geometry.",
            )
            relationship_warnings: list[str] = []
            for context in page_contracts:
                relationship_warnings.extend(validate_vlm_relationships(
                    context["result"],
                    context["ocr_contract"],
                    context["layout_contract"],
                ))
            record["relationship_warnings"] = list(dict.fromkeys(relationship_warnings))
            record["vlm_results"] = vlm_results
            record["vlm_result"] = vlm_results[0]
            record["draft"] = self._build_multi_page_draft(record, page_contracts)
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
    ) -> list[tuple[bytes, ImageIdentity]] | None:
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
        if not isinstance(summary.get("page_count"), int) or summary["page_count"] < 1:
            raise AdapterError("Visual-field preprocessing returned no template pages")

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

        canonical_pages: list[tuple[bytes, ImageIdentity]] = []
        identity_records: list[dict[str, Any]] = []
        artifact_records: list[dict[str, Any]] = []
        manifest_pages = preprocess_manifest.get("pages") or []
        if len(manifest_pages) != summary["page_count"]:
            raise AdapterError("Preprocessing summary and manifest page counts do not match")
        for expected_number, manifest_page in enumerate(manifest_pages, 1):
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
            if page_id != f"page_{expected_number:03d}" or page_number != expected_number:
                raise AdapterError("Preprocessing manifest has invalid or non-sequential page identities")
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
            canonical_path = (
                self.settings.storage_root / record["id"] / f"page_{page_number:03d}.png"
            )
            canonical_path.write_bytes(page_bytes)
            identity_record = {
                "sha256": identity.sha256,
                "width": identity.width,
                "height": identity.height,
                "document_id": identity.document_id,
                "page_id": identity.page_id,
                "page_number": identity.page_number,
            }
            canonical_pages.append((page_bytes, identity))
            identity_records.append(identity_record)
            artifact_records.append({
                "page_id": page_id,
                "page_number": page_number,
                "upstream_path": image_path,
                "retrieved_at": iso_now(),
            })
        record["image_identities"] = identity_records
        record["image_identity"] = identity_records[0]
        record["page_images"] = [
            f"/api/v1/template-registrations/{record['id']}/pages/{page_number}"
            for page_number in range(1, len(identity_records) + 1)
        ]
        record["canonical_artifacts"] = artifact_records
        record["canonical_artifact"] = artifact_records[0]
        record["updated_at"] = iso_now()
        self.store.put("registration", record["id"], record)
        self.store.add_audit(
            action="established canonical template page",
            target_type="template_registration",
            target_id=record["id"],
            correlation_id=correlation_id,
            after={"page_count": len(identity_records), "pages": identity_records},
        )
        return canonical_pages

    @staticmethod
    def _build_preprocessing_result(
        requested_policy: str,
        correction_mode: str,
        manifest: dict[str, Any],
    ) -> dict[str, Any]:
        pages = manifest.get("pages") or []
        if not pages:
            raise AdapterError("Preprocessing manifest contains no pages")
        if manifest.get("page_count") != len(pages):
            raise AdapterError("Preprocessing manifest page_count is inconsistent")

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
        identity: ImageIdentity,
        correlation_id: str,
    ) -> dict[str, Any]:
        try:
            result = (
                await self.client.request(
                    "ocr-fastapi-service",
                    "POST",
                    f"{self.settings.ocr_url}/v1/ocr/process",
                    correlation_id=correlation_id,
                    params={
                        "preprocess_mode": "minimal",
                        "language": "eng+mya",
                        "document_id": record["downstream_ids"]["visual_document_id"],
                        "page_number": identity.page_number,
                    },
                    files={
                        "file": (f"{identity.page_id}.png", page_bytes, "image/png")
                    },
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
        self,
        record: dict[str, Any],
        result: dict[str, Any],
        layout_contract: dict[str, Any],
        identity_record: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        semantic = result.get("semantic_output") or {}
        pages = semantic.get("pages") or []
        semantic_page = pages[0] if pages else {}
        labels = {item.get("label_id"): item for item in semantic_page.get("semantic_labels", [])}
        coverage_records = {
            item.get("region_id"): item
            for item in (result.get("coverage_output") or {}).get("records", [])
        }
        fields_by_region: dict[str, dict[str, Any]] = {}
        for field in semantic_page.get("fields", []):
            for region_id in field.get("region_ids", []):
                fields_by_region[region_id] = field
        identity = identity_record or record["image_identity"]
        page_number = int(identity["page_number"])
        draft_regions: list[dict[str, Any]] = []
        for index, layout_region in enumerate(layout_contract["pages"][0]["regions"], 1):
            if layout_region["region_type"] == "TABLE_CELL":
                continue
            semantic_field = fields_by_region.get(layout_region["region_id"])
            label = labels.get((semantic_field or {}).get("label_id"), {})
            field_type = str((semantic_field or {}).get("field_type", ""))
            extraction_mode = VLM_FIELD_TYPES.get(field_type) or LAYOUT_FALLBACK_MODES.get(layout_region["region_type"])
            semantic_option = next(
                (
                    option
                    for option in (semantic_field or {}).get("options", [])
                    if option.get("control_region_id") == layout_region["region_id"]
                ),
                None,
            )
            field_confidence = float((semantic_field or {}).get("confidence", layout_region.get("confidence", 0)))
            review_reasons = []
            if semantic_field is None:
                review_reasons.append("No semantic field mapping was found")
            if extraction_mode is None:
                review_reasons.append(f"Unsupported or ambiguous field type: {field_type or layout_region['region_type']}")
            if field_confidence < 0.8:
                review_reasons.append("Low AI confidence")
            if (semantic_field or {}).get("review_required"):
                review_reasons.extend(
                    str(item)
                    for item in (semantic_field or {}).get("review_reasons", [])
                    if isinstance(item, str) and item.strip()
                )
                if not (semantic_field or {}).get("review_reasons"):
                    review_reasons.append("Model requested human review")
            review_required = bool(review_reasons)
            semantic_key = (semantic_field or {}).get("key") or f"field_{index:03d}"
            option_key = (semantic_option or {}).get("option_key")
            key = f"{semantic_key}_{option_key}" if option_key else semantic_key
            semantic_field_id = (semantic_field or {}).get("field_id") or key
            field_id = f"{semantic_field_id}_{option_key}" if option_key else semantic_field_id
            primary_label = label.get("primary_text") or semantic_key.replace("_", " ").title()
            option_value = (semantic_option or {}).get("value")
            display_label = f"{primary_label}: {option_value}" if option_value else primary_label
            draft_regions.append(
                {
                    "id": layout_region["region_id"],
                    "region_type": layout_region["region_type"],
                    "field_id": field_id,
                    "page": page_number,
                    "key": key,
                    "label": display_label,
                    "data_type": field_type or None,
                    "language": label.get("primary_language") or "unknown",
                    "extraction_mode": extraction_mode,
                    "required": bool((semantic_field or {}).get("required", False)),
                    "confidence": field_confidence,
                    "bbox": xyxy_to_normalized_xywh(layout_region["bbox_px"], identity["width"], identity["height"]),
                    "source_region_ids": [layout_region["region_id"]],
                    "source_label_id": (semantic_field or {}).get("label_id"),
                    "label_token_ids": list(label.get("token_ids") or []),
                    "relationship": (semantic_field or {}).get("relationship"),
                    "semantic_group_field_id": (semantic_field or {}).get("field_id") if semantic_option else None,
                    "option_key": option_key,
                    "parent_region_id": layout_region.get("parent_region_id"),
                    "review_required": review_required,
                    "review_reasons": review_reasons,
                    "model_metadata": (semantic_field or {}).get("model_metadata"),
                    "validation": (semantic_field or {}).get("validation"),
                    "enabled": True,
                    "geometry_source": "PP-DocLayoutV3",
                }
            )
        structural_regions = [
            {
                "id": region["region_id"],
                "page": page_number,
                "region_type": region["region_type"],
                "parent_region_id": region.get("parent_region_id"),
                "bbox": xyxy_to_normalized_xywh(
                    region["bbox_px"], identity["width"], identity["height"]
                ),
                "geometry_source": "PP-DocLayoutV3",
            }
            for region in layout_contract["pages"][0]["regions"]
            if region["region_type"] == "TABLE_CELL"
        ]
        unassigned_regions = [
            {
                "region_id": region_id,
                "page": page_number,
                "region_type": coverage.get("region_type"),
                "bbox": xyxy_to_normalized_xywh(
                    coverage["bbox_px"], identity["width"], identity["height"]
                ),
                "status": coverage.get("status"),
                "needs_review": coverage.get("needs_review"),
            }
            for region_id, coverage in coverage_records.items()
            if coverage.get("status") == "REVIEW_REQUIRED"
        ]
        return {
            "schema_version": "1.0.0",
            "revision": 1,
            "page": {
                "page_id": identity["page_id"],
                "page_number": identity["page_number"],
                "image_url": (
                    f"/api/v1/template-registrations/{record['id']}/pages/{page_number}"
                ),
                "width": identity["width"],
                "height": identity["height"],
                "sha256": identity["sha256"],
            },
            "regions": draft_regions,
            "structural_regions": structural_regions,
            "unassigned_regions": unassigned_regions,
            "warnings": (
                semantic.get("warnings", [])
                + record.get("adapter_warnings", [])
                + record.get("relationship_warnings", [])
            ),
            "quality_summary": result.get("quality_summary", {}),
        }

    def _build_multi_page_draft(
        self,
        record: dict[str, Any],
        page_contracts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        page_drafts = [
            self._build_editable_draft(
                record,
                context["result"],
                context["layout_contract"],
                {
                    "sha256": context["identity"].sha256,
                    "width": context["identity"].width,
                    "height": context["identity"].height,
                    "document_id": context["identity"].document_id,
                    "page_id": context["identity"].page_id,
                    "page_number": context["identity"].page_number,
                },
            )
            for context in page_contracts
        ]
        regions: list[dict[str, Any]] = []
        seen_keys: set[str] = set()
        seen_field_ids: set[str] = set()
        for page_draft in page_drafts:
            for region in page_draft["regions"]:
                item = dict(region)
                page_number = int(item["page"])
                key = str(item["key"])
                if key in seen_keys:
                    candidate = f"{key}_page_{page_number}"
                    suffix = 2
                    while candidate in seen_keys:
                        candidate = f"{key}_page_{page_number}_{suffix}"
                        suffix += 1
                    item["key"] = candidate
                seen_keys.add(str(item["key"]))
                field_id = str(item["field_id"])
                if field_id in seen_field_ids:
                    candidate = f"{field_id}_page_{page_number}"
                    suffix = 2
                    while candidate in seen_field_ids:
                        candidate = f"{field_id}_page_{page_number}_{suffix}"
                        suffix += 1
                    item["field_id"] = candidate
                seen_field_ids.add(str(item["field_id"]))
                regions.append(item)

        warnings: list[Any] = []
        warning_keys: set[str] = set()
        for item in (
            warning
            for page_draft in page_drafts
            for warning in page_draft.get("warnings", [])
        ):
            key = json.dumps(item, sort_keys=True, ensure_ascii=False, default=str)
            if key not in warning_keys:
                warning_keys.add(key)
                warnings.append(item)

        quality_summaries = [page.get("quality_summary") or {} for page in page_drafts]
        count_fields = (
            "target_region_count",
            "semantic_field_count",
            "assigned_region_count",
            "assigned_review_region_count",
            "unassigned_region_count",
            "structural_region_count",
            "semantic_consistency_warning_count",
            "structured_table_count",
        )
        quality_summary = {
            field: sum(int(summary.get(field, 0)) for summary in quality_summaries)
            for field in count_fields
        }
        target_count = quality_summary["target_region_count"]
        quality_summary.update({
            "actionable_coverage_ratio": round(
                quality_summary["assigned_region_count"] / max(1, target_count), 6
            ),
            "mapping_complete": all(
                bool(summary.get("mapping_complete")) for summary in quality_summaries
            ),
            "automation_ready": False,
            "page_count": len(page_drafts),
        })
        quality_summary["quality_status"] = (
            "INCOMPLETE_REVIEW_REQUIRED"
            if not quality_summary["mapping_complete"]
            else "MAPPED_REVIEW_REQUIRED"
            if quality_summary["assigned_review_region_count"]
            else "MAPPED"
        )
        return {
            "schema_version": "1.0.0",
            "revision": 1,
            "page": page_drafts[0]["page"],
            "pages": [page["page"] for page in page_drafts],
            "regions": regions,
            "structural_regions": [
                item
                for page_draft in page_drafts
                for item in page_draft["structural_regions"]
            ],
            "unassigned_regions": [
                item
                for page_draft in page_drafts
                for item in page_draft["unassigned_regions"]
            ],
            "warnings": warnings,
            "quality_summary": quality_summary,
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
        identities = record.get("image_identities") or [record["image_identity"]]
        identity = identities[0]
        definition, review_flags = semantic_draft_to_template(
            template_id=pinned_id,
            name=record.get("name") or f"{record['form_type_id'].title()} Claim Template",
            width=identity["width"],
            height=identity["height"],
            regions=record["draft"]["regions"],
            page_dimensions=(
                [
                    {
                        "page_number": item["page_number"],
                        "width": item["width"],
                        "height": item["height"],
                    }
                    for item in identities
                ]
                if len(identities) > 1
                else None
            ),
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
        reference_paths: list[Path] = []
        for page_number in range(1, len(identities) + 1):
            reference_path = self.settings.storage_root / registration_id / f"page_{page_number:03d}.png"
            if not reference_path.is_file():
                raise AdapterError(f"Approved template is missing canonical reference page {page_number}")
            await self.client.request(
                "document-processing-layer",
                "POST",
                f"{self.settings.document_processing_url}/api/v1/templates/{pinned_id}/reference",
                correlation_id=correlation_id,
                params={"page_number": page_number},
                files={"file": (reference_path.name, reference_path.read_bytes(), "image/png")},
            )
            reference_paths.append(reference_path)
        now = iso_now()
        version_id = f"{template_id}:v{revision}"
        version = {
            "id": version_id,
            "template_id": template_id,
            "version": str(revision),
            "definition": registered_definition,
            "draft_snapshot": record["draft"],
            "reference_image_path": str(reference_paths[0]),
            "reference_image_paths": [str(path) for path in reference_paths],
            "approved_at": now,
            "approved_by": actor,
            "immutable": True,
        }
        self.store.put("template_version", version_id, version, create_only=True)
        template = {
            "id": template_id,
            "name": record.get("name") or f"{record['form_type_id'].title()} Claim Template",
            "description": record.get("description", ""),
            "form_type_id": record["form_type_id"],
            "version": str(revision),
            "version_id": version_id,
            "downstream_template_id": pinned_id,
            "status": "active",
            "confidence_score": record.get("draft", {}).get("quality_summary", {}).get("actionable_coverage_ratio", 0),
            "pages": len(identities),
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

    async def match_document_template(
        self, *, content: bytes, filename: str, correlation_id: str
    ) -> dict[str, Any]:
        """Register approved references downstream and return an evidence-based match."""
        templates = [
            item for item in self.store.list("template")
            if item.get("status") == "active" and not item.get("deleted_at")
        ]
        for template in templates:
            version = self.store.require("template_version", template["version_id"])
            definition = version.get("definition")
            if not isinstance(definition, dict):
                continue
            await self.client.request(
                "document-processing-layer", "POST",
                f"{self.settings.document_processing_url}/api/v1/templates/register",
                correlation_id=correlation_id, json=definition,
            )
            reference_paths = version.get("reference_image_paths") or [version.get("reference_image_path")]
            expected_pages = int(template.get("pages") or 1)
            if len(reference_paths) < expected_pages and isinstance(reference_paths[0] if reference_paths else None, str):
                first_path = Path(reference_paths[0])
                recovered_paths = [
                    str(first_path.parent / f"page_{page_number:03d}.png")
                    for page_number in range(1, expected_pages + 1)
                ]
                if all(Path(path).is_file() for path in recovered_paths):
                    reference_paths = recovered_paths
            for page_number, reference_path_value in enumerate(reference_paths[:expected_pages], 1):
                if not isinstance(reference_path_value, str) or not Path(reference_path_value).is_file():
                    continue
                reference_path = Path(reference_path_value)
                await self.client.request(
                    "document-processing-layer", "POST",
                    f"{self.settings.document_processing_url}/api/v1/templates/{template['downstream_template_id']}/reference",
                    correlation_id=correlation_id, params={"page_number": page_number},
                    files={"file": (reference_path.name, reference_path.read_bytes(), "image/png")},
                )
        response = await self.client.request(
            "document-processing-layer", "POST",
            f"{self.settings.document_processing_url}/api/v1/documents/match-template",
            correlation_id=correlation_id,
            files={"file": (filename, content, mimetypes.guess_type(filename)[0] or "application/octet-stream")},
        )
        return response.json()

    async def _preprocess_completed_document(
        self, record: dict[str, Any], correlation_id: str
    ) -> list[tuple[bytes, ImageIdentity]] | None:
        """Create paper-cropped canonical pages for a completed form.

        This deliberately uses the same visual preprocessing contract as template
        registration.  OCR receives these pages directly, so it does not need to
        feature-warp the user's photo to the template a second time.
        """
        source_path = Path(record["source_path"])
        response = await self.client.request(
            "visual-field-detection",
            "POST",
            f"{self.settings.visual_field_url}/v1/documents",
            correlation_id=correlation_id,
            files={
                "file": (
                    record["file_name"],
                    source_path.read_bytes(),
                    mimetypes.guess_type(record["file_name"])[0] or "application/octet-stream",
                )
            },
        )
        visual_id = response.json()["document_id"]
        record["downstream_ids"]["visual_document_id"] = visual_id

        response = await self.client.request(
            "visual-field-detection",
            "POST",
            f"{self.settings.visual_field_url}/v1/documents/{visual_id}/preprocess",
            correlation_id=correlation_id,
            json={
                "correction_mode": "auto",
                "capture_profile": "document",
                "deskew": True,
                "normalize_illumination": True,
                "sharpen": True,
            },
        )
        preprocess_result = response.json()
        artifact_path = str(preprocess_result.get("artifact") or "")
        if not artifact_path or artifact_path.startswith("/") or ".." in Path(artifact_path).parts:
            raise AdapterError("Visual-field preprocessing did not return a valid manifest artifact")
        manifest = (
            await self.client.request(
                "visual-field-detection",
                "GET",
                f"{self.settings.visual_field_url}/v1/documents/{visual_id}/artifacts/{artifact_path}",
                correlation_id=correlation_id,
            )
        ).json()
        preprocessing = self._build_preprocessing_result("auto", "auto", manifest)
        if preprocessing["capture_profile"] != "document":
            raise AdapterError("Visual-field preprocessing returned the wrong capture profile")
        record["preprocessing"] = preprocessing

        if preprocessing["retake_required"]:
            record["status"] = "needs_review"
            record["progress"] = {
                "stage": "capture_quality",
                "percent": 100,
                "message": "The uploaded photo could not be safely paper-cropped. Please retake it.",
            }
            record["updated_at"] = iso_now()
            self.store.put("document", record["id"], record)
            self.store.add_audit(
                action="document requires recapture",
                target_type="document",
                target_id=record["id"],
                correlation_id=correlation_id,
                after={"reasons": preprocessing["reasons"]},
            )
            return None

        pages = manifest.get("pages") or []
        if len(pages) != preprocessing["page_count"]:
            raise AdapterError("Preprocessing manifest page count is inconsistent")
        canonical_pages: list[tuple[bytes, ImageIdentity]] = []
        identity_records: list[dict[str, Any]] = []
        for expected_number, manifest_page in enumerate(pages, 1):
            image_path = str(manifest_page.get("image_path") or "")
            if not image_path or image_path.startswith("/") or ".." in Path(image_path).parts:
                raise AdapterError("Preprocessing manifest did not identify a valid canonical page artifact")
            page_bytes = (
                await self.client.request(
                    "visual-field-detection",
                    "GET",
                    f"{self.settings.visual_field_url}/v1/documents/{visual_id}/artifacts/{image_path}",
                    correlation_id=correlation_id,
                )
            ).content
            page_id = str(manifest_page.get("page_id") or "")
            page_number = manifest_page.get("page_number")
            if not page_bytes or page_id != f"page_{expected_number:03d}" or page_number != expected_number:
                raise AdapterError("Preprocessing manifest has invalid canonical page identities")
            identity = ImageIdentity.from_bytes(
                page_bytes, document_id=visual_id, page_id=page_id, page_number=page_number
            )
            if manifest_page.get("sha256") != identity.sha256:
                raise AdapterError("Canonical document page SHA-256 does not match the preprocessing manifest")
            if manifest_page.get("width") != identity.width or manifest_page.get("height") != identity.height:
                raise AdapterError("Canonical document page dimensions do not match the preprocessing manifest")
            canonical_path = self.settings.storage_root / record["id"] / f"canonical_page_{page_number:03d}.png"
            canonical_path.write_bytes(page_bytes)
            canonical_pages.append((page_bytes, identity))
            identity_records.append({
                "sha256": identity.sha256,
                "width": identity.width,
                "height": identity.height,
                "document_id": identity.document_id,
                "page_id": identity.page_id,
                "page_number": identity.page_number,
                "path": str(canonical_path),
            })
        record["canonical_pages"] = identity_records
        record["updated_at"] = iso_now()
        self.store.put("document", record["id"], record)
        return canonical_pages

    @staticmethod
    def _canonical_document_payload(
        pages: list[tuple[bytes, ImageIdentity]]
    ) -> tuple[bytes, str, str]:
        """Return the canonical pages as the image/PDF shape expected by OCR."""
        if len(pages) == 1:
            return pages[0][0], "canonical-page-001.png", "image/png"
        rendered_pages: list[Image.Image] = []
        for page_bytes, _ in pages:
            with Image.open(io.BytesIO(page_bytes)) as page:
                rendered_pages.append(page.convert("RGB"))
        output = io.BytesIO()
        rendered_pages[0].save(
            output, format="PDF", save_all=True, append_images=rendered_pages[1:], resolution=144.0
        )
        return output.getvalue(), "canonical-document.pdf", "application/pdf"

    async def _match_canonical_document(
        self, record: dict[str, Any], content: bytes, filename: str, correlation_id: str
    ) -> bool:
        result = await self.match_document_template(
            content=content, filename=filename, correlation_id=correlation_id
        )
        downstream_template_id = result.get("selected_template_id")
        template = next(
            (
                item for item in self.store.list("template")
                if item.get("downstream_template_id") == downstream_template_id
                and item.get("status") == "active" and not item.get("deleted_at")
            ),
            None,
        )
        candidates = [
            {
                **candidate,
                "template_id": next(
                    (
                        item["id"] for item in self.store.list("template")
                        if item.get("downstream_template_id") == candidate.get("template_id")
                    ),
                    candidate.get("template_id"),
                ),
            }
            for candidate in result.get("candidates", [])
        ]
        if template is None:
            record["template_match"] = {
                "template_id": None,
                "score": float(result.get("score") or 0),
                "confirmed": False,
                "reason": result.get("reason") or "Template match is uncertain; select a template to continue.",
                "candidates": candidates,
            }
            record["status"] = "needs_review"
            record["progress"] = {"stage": "template_match", "percent": 100}
            record["updated_at"] = iso_now()
            self.store.put("document", record["id"], record)
            return False
        record["template_id"] = template["id"]
        record["template_version"] = template["version_id"]
        record["form_type_id"] = template["form_type_id"]
        record["pages"] = int(template.get("pages") or 1)
        record["template_match"] = {
            "template_id": template["id"],
            "version": template["version"],
            "score": float(result.get("score") or 0),
            "confirmed": True,
            "reason": "Automatically matched approved template",
            "candidates": candidates,
        }
        record["updated_at"] = iso_now()
        self.store.put("document", record["id"], record)
        return True

    async def run_document(self, document_id: str) -> None:
        record = self.store.require("document", document_id)
        correlation_id = record["correlation_id"]
        try:
            record["status"] = "processing"
            record["progress"] = {"stage": "paper_detection", "percent": 15}
            record["updated_at"] = iso_now()
            self.store.put("document", document_id, record)
            canonical_pages = await self._preprocess_completed_document(record, correlation_id)
            if canonical_pages is None:
                return
            canonical_content, canonical_filename, canonical_media_type = self._canonical_document_payload(canonical_pages)
            if not record.get("template_id") and not await self._match_canonical_document(
                record, canonical_content, canonical_filename, correlation_id
            ):
                return
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
            reference_image_paths = template_version.get("reference_image_paths") or [
                template_version.get("reference_image_path")
            ]
            expected_pages = int(template.get("pages") or 1)
            if len(reference_image_paths) < expected_pages:
                # Versions approved before multi-page references were introduced
                # still point at page_001. Reconstruct the sibling canonical paths
                # when the immutable registration artifacts are still present.
                first_reference = reference_image_paths[0] if reference_image_paths else None
                if isinstance(first_reference, str):
                    first_path = Path(first_reference)
                    recovered_paths = [
                        str(first_path.parent / f"page_{page_number:03d}.png")
                        for page_number in range(1, expected_pages + 1)
                    ]
                    if all(Path(path).is_file() for path in recovered_paths):
                        reference_image_paths = recovered_paths
                if len(reference_image_paths) < expected_pages:
                    raise AdapterError("approved template version is missing canonical reference pages")
            for page_number, reference_image_path in enumerate(reference_image_paths[:expected_pages], 1):
                if not isinstance(reference_image_path, str) or not Path(reference_image_path).is_file():
                    raise AdapterError(f"approved template version is missing canonical reference page {page_number}")
                reference_path = Path(reference_image_path)
                await self.client.request(
                    "document-processing-layer",
                    "POST",
                    f"{self.settings.document_processing_url}/api/v1/templates/{template['downstream_template_id']}/reference",
                    correlation_id=correlation_id,
                    params={"page_number": page_number},
                    files={"file": (reference_path.name, reference_path.read_bytes(), "image/png")},
                )
            record["status"] = "processing"
            record["progress"] = {"stage": "document_processing", "percent": 35}
            record["updated_at"] = iso_now()
            self.store.put("document", document_id, record)
            response = await self.client.request(
                "document-processing-layer",
                "POST",
                f"{self.settings.document_processing_url}/api/v1/documents/process",
                correlation_id=correlation_id,
                files={"file": (canonical_filename, canonical_content, canonical_media_type)},
                data={"template_id": template["downstream_template_id"], "canonicalized_pages": "true"},
            )
            upstream = response.json()
            record["downstream_ids"]["document_job_id"] = upstream["job_id"]
            record["pages"] = int(upstream.get("page_count") or record.get("pages") or 1)
            record["processed"] = self._adapt_document_result(upstream)
            record["extraction_attempts"].append(
                {"attempt": len(record["extraction_attempts"]) + 1, "created_at": iso_now(), "upstream": upstream}
            )
            if upstream.get("status") == "FAILED":
                record["status"] = "needs_review"
                record["failure"] = {
                    "code": "DOCUMENT_ALIGNMENT_FAILED",
                    "message": upstream.get("error") or "Document alignment failed.",
                }
            else:
                record["status"] = "needs_review" if upstream.get("needs_human_review", False) else "ready_to_sync"
                # A retry can recover from an earlier alignment failure.  Do
                # not leave that historical failure on a successful partial
                # or complete extraction result.
                record.pop("failure", None)
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
                "preprocessed_source_region": item.get("preprocessed_crop_path"),
                "line_source_regions": item.get("line_crop_paths", []),
                "ocr_mode": item.get("ocr_mode", "full_field_fallback"),
                "field_type": item.get("field_type", "printed_text"),
                "is_valid": bool(item.get("validation_passed", False)),
                "validation_status": "valid" if item.get("validation_passed", False) else "invalid",
                "review_status": "required" if item.get("human_review_flag", False) else "not_required",
                "requires_review": bool(item.get("human_review_flag", False)),
                "errors": errors,
                "warnings": warnings,
                "input_field": key,
                "label": item.get("label", key.replace("_", " ").title()),
                "page": int(item.get("page") or 1),
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
                "page_count": int(upstream.get("page_count") or 1),
                "page_alignment_scores": upstream.get("page_alignment_scores") or {},
                "aligned_page_count": len(upstream.get("aligned_page_paths") or {}),
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
