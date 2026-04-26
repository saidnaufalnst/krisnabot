from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=2000)


class ChatResponse(BaseModel):
    answer: str
    message: str = ""
    found: bool
    used_files: list[str]
    error: str = ""
    grounding_chunks: list[dict[str, str]] = Field(default_factory=list)


class IngestResponse(BaseModel):
    running: bool
    status: str
    message: str
    force: bool = False
    target_files: list[str] = []
    uploaded: int
    skipped: int
    failed: int
    failed_files: list[dict[str, str]]
    indexed_files: int
    total_files: int
    processed_files: int
    current_file: str
    current_chunk: int = 0
    current_file_total_chunks: int = 0
    resumed_from_chunk: int = 0


class UploadResponse(BaseModel):
    saved_files: list[str]
    total_files_in_docs: int


class AdminStatusResponse(BaseModel):
    docs_files: list[str]
    indexed_files: list[str]


class FileActionResponse(BaseModel):
    message: str
    docs_files: list[str]
    indexed_files: list[str]
