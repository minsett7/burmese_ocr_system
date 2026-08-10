from __future__ import annotations

import copy
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select

from .database import Base, Record


def iso_now() -> str:
    return datetime.now(UTC).isoformat()


class RecordStore:
    def __init__(self, engine, session_factory):
        self.engine = engine
        self.sessions = session_factory

    def initialize(self) -> None:
        Base.metadata.create_all(self.engine)

    def put(self, kind: str, record_id: str, data: dict[str, Any], *, create_only: bool = False) -> dict[str, Any]:
        payload = copy.deepcopy(data)
        with self.sessions.begin() as session:
            record = session.get(Record, (kind, record_id))
            if record is None:
                record = Record(kind=kind, id=record_id, data=payload)
                session.add(record)
            elif create_only:
                raise ValueError(f"{kind} record already exists: {record_id}")
            else:
                record.data = payload
                record.updated_at = datetime.now(UTC)
        return payload

    def get(self, kind: str, record_id: str) -> dict[str, Any] | None:
        with self.sessions() as session:
            record = session.get(Record, (kind, record_id))
            return copy.deepcopy(record.data) if record else None

    def require(self, kind: str, record_id: str) -> dict[str, Any]:
        value = self.get(kind, record_id)
        if value is None:
            raise KeyError(record_id)
        return value

    def list(self, kind: str) -> list[dict[str, Any]]:
        with self.sessions() as session:
            records = session.scalars(
                select(Record).where(Record.kind == kind).order_by(Record.created_at.desc())
            ).all()
            return [copy.deepcopy(record.data) for record in records]

    def add_audit(
        self,
        *,
        action: str,
        target_type: str,
        target_id: str,
        actor: str = "system",
        before: Any = None,
        after: Any = None,
        correlation_id: str | None = None,
        template_version: str | None = None,
        extraction_attempt: int | None = None,
    ) -> dict[str, Any]:
        event_id = f"AUD-{uuid.uuid4().hex[:12].upper()}"
        event = {
            "id": event_id,
            "actor": actor,
            "actor_id": actor,
            "actor_display_name": actor,
            "action": action,
            "target_type": target_type,
            "target_id": target_id,
            "before": before,
            "after": after,
            "correlation_id": correlation_id,
            "template_version": template_version,
            "extraction_attempt": extraction_attempt,
            "created_at": iso_now(),
        }
        return self.put("audit", event_id, event, create_only=True)
