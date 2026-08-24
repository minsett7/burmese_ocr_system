from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


@dataclass(frozen=True, slots=True)
class Settings:
    database_url: str
    storage_root: Path
    document_processing_url: str
    visual_field_url: str
    ocr_url: str
    vlm_url: str
    vlm_api_key: str | None
    request_timeout_seconds: float
    retry_attempts: int
    retry_backoff_seconds: float
    poll_interval_seconds: float
    poll_timeout_seconds: float
    max_upload_mb: int
    allowed_extensions: tuple[str, ...]
    cors_origins: tuple[str, ...]

    @classmethod
    def from_env(cls) -> "Settings":
        value = cls(
            database_url=os.getenv("DATABASE_URL", "sqlite:///./runtime/orchestrator.db"),
            storage_root=Path(os.getenv("ORCHESTRATOR_STORAGE_ROOT", "runtime/artifacts")).resolve(),
            document_processing_url=os.getenv("DOCUMENT_PROCESSING_URL", "http://document-processing-layer:8000").rstrip("/"),
            visual_field_url=os.getenv("VISUAL_FIELD_URL", "http://visual-field-detection:8000").rstrip("/"),
            ocr_url=os.getenv("OCR_URL", "http://ocr-fastapi-service:8000").rstrip("/"),
            vlm_url=os.getenv("INSURANCE_VLM_URL", "http://insurance-vlm:8000").rstrip("/"),
            vlm_api_key=os.getenv("VLM_API_KEY") or None,
            request_timeout_seconds=float(os.getenv("REQUEST_TIMEOUT_SECONDS", "300")),
            retry_attempts=int(os.getenv("RETRY_ATTEMPTS", "3")),
            retry_backoff_seconds=float(os.getenv("RETRY_BACKOFF_SECONDS", "0.25")),
            poll_interval_seconds=float(os.getenv("VLM_POLL_INTERVAL_SECONDS", "0.25")),
            poll_timeout_seconds=float(os.getenv("VLM_POLL_TIMEOUT_SECONDS", "120")),
            max_upload_mb=int(os.getenv("MAX_UPLOAD_MB", "25")),
            allowed_extensions=_csv(os.getenv("ALLOWED_UPLOAD_EXTENSIONS", ".pdf,.png,.jpg,.jpeg,.webp,.tif,.tiff")),
            cors_origins=_csv(os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173,http://127.0.0.1:5173")),
        )
        if value.max_upload_mb < 1 or value.retry_attempts < 1:
            raise ValueError("MAX_UPLOAD_MB and RETRY_ATTEMPTS must be positive")
        if value.request_timeout_seconds <= 0 or value.poll_timeout_seconds <= 0:
            raise ValueError("Timeouts must be positive")
        value.storage_root.mkdir(parents=True, exist_ok=True)
        return value

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024
