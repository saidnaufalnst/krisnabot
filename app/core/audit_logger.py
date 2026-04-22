from __future__ import annotations

from typing import Any

from app.repositories.audit_repository import AuditRepository


class AuditLogger:
    def __init__(self, event_type: str) -> None:
        self.event_type = event_type
        self.repository = AuditRepository()

    def write(self, payload: dict[str, Any]) -> None:
        self.repository.create(self.event_type, payload)
