from __future__ import annotations

from typing import Any

from app.db.models import AuditLog
from app.db.session import SessionLocal


class AuditRepository:
    def create(self, event_type: str, payload: dict[str, Any]) -> None:
        with SessionLocal() as session:
            session.add(AuditLog(event_type=event_type, payload=payload))
            session.commit()
