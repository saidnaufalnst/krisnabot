from __future__ import annotations

from app.db.models import ChatLog
from app.db.session import SessionLocal


class ChatLogRepository:
    def create(self, *, question: str, answer: str, found: bool, used_files: list[str]) -> None:
        with SessionLocal() as session:
            session.add(
                ChatLog(
                    question=question,
                    answer=answer,
                    found=found,
                    used_files=used_files,
                )
            )
            session.commit()
