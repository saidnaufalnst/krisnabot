import logging
from functools import lru_cache

from fastapi import APIRouter, HTTPException

from app.schemas import ChatRequest, ChatResponse
from app.services.rag_service import RAGService

router = APIRouter(prefix="/user", tags=["user"])
logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_rag_service() -> RAGService:
    return RAGService()


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    try:
        rag_service = get_rag_service()
        result = rag_service.ask(payload.question)
        return ChatResponse(**result)
    except Exception as exc:
        logger.exception("Chat failed")
        raise HTTPException(status_code=500, detail=f"Chat gagal diproses: {exc}") from exc
