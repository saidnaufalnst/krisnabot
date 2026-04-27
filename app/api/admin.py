import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile

from app.core.service_auth import require_admin_service_key
from app.schemas import AdminStatusResponse, FileActionResponse, IngestResponse, UploadResponse
from app.services.ingest_job_manager import ingest_job_manager
from app.services.ingestion_service import IngestionService

router = APIRouter(
    prefix="/admin",
    tags=["admin"],
    dependencies=[Depends(require_admin_service_key)],
)
logger = logging.getLogger(__name__)


def get_ingestion_service() -> IngestionService:
    return IngestionService()


@router.get("/status", response_model=AdminStatusResponse)
def admin_status() -> AdminStatusResponse:
    ingestion_service = get_ingestion_service()
    return AdminStatusResponse(
        docs_files=ingestion_service.list_docs(),
        indexed_files=ingestion_service.list_indexed_files(),
    )


@router.post("/upload", response_model=UploadResponse)
async def upload_docs(
    files: list[UploadFile] | None = File(None),
    file: UploadFile | None = File(None),
) -> UploadResponse:
    try:
        ingestion_service = get_ingestion_service()
        payloads: list[tuple[str, bytes]] = []
        uploaded_files = list(files or [])
        if file is not None:
            uploaded_files.append(file)
        if not uploaded_files:
            raise HTTPException(
                status_code=422,
                detail="Field upload wajib diisi pada multipart form dengan nama 'files' atau 'file'.",
            )

        for uploaded_file in uploaded_files:
            if uploaded_file.content_type != "application/pdf":
                raise HTTPException(
                    status_code=400,
                    detail=f"File '{uploaded_file.filename}' harus berformat PDF.",
                )
            payloads.append((uploaded_file.filename or "document.pdf", await uploaded_file.read()))

        saved_files = ingestion_service.save_files(payloads)
        return UploadResponse(
            saved_files=saved_files,
            total_files_in_docs=len(ingestion_service.list_docs()),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Upload failed")
        raise HTTPException(status_code=500, detail=f"Upload dokumen gagal: {exc}") from exc


@router.post("/replace", response_model=FileActionResponse)
async def replace_doc(
    source_file: str = Form(...),
    file: UploadFile = File(...),
) -> FileActionResponse:
    try:
        ingestion_service = get_ingestion_service()
        if file.content_type != "application/pdf":
            raise HTTPException(
                status_code=400,
                detail=f"File '{file.filename}' harus berformat PDF.",
            )
        target_name = ingestion_service.replace_file(
            source_file=source_file,
            filename=file.filename or source_file,
            content=await file.read(),
        )
        return FileActionResponse(
            message=f"File '{source_file}' berhasil diganti menjadi '{target_name}'. Jalankan ingest ulang untuk memperbarui index.",
            docs_files=ingestion_service.list_docs(),
            indexed_files=ingestion_service.list_indexed_files(),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Replace failed")
        raise HTTPException(status_code=500, detail=f"Gagal mengganti dokumen: {exc}") from exc


@router.delete("/docs", response_model=FileActionResponse)
def delete_doc(source_file: list[str] = Query(...)) -> FileActionResponse:
    try:
        ingestion_service = get_ingestion_service()
        deleted_files: list[str] = []
        for item in source_file:
            ingestion_service.delete_file(item)
            deleted_files.append(item)
        return FileActionResponse(
            message=f"{len(deleted_files)} file berhasil dihapus: {', '.join(deleted_files)}",
            docs_files=ingestion_service.list_docs(),
            indexed_files=ingestion_service.list_indexed_files(),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Delete failed")
        raise HTTPException(status_code=500, detail=f"Gagal menghapus dokumen: {exc}") from exc


@router.delete("/index", response_model=FileActionResponse)
def delete_index(source_file: list[str] = Query(...)) -> FileActionResponse:
    try:
        ingestion_service = get_ingestion_service()
        cleared_files: list[str] = []
        for item in source_file:
            ingestion_service.delete_index(item)
            cleared_files.append(item)
        return FileActionResponse(
            message=f"{len(cleared_files)} index berhasil dihapus: {', '.join(cleared_files)}",
            docs_files=ingestion_service.list_docs(),
            indexed_files=ingestion_service.list_indexed_files(),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Delete index failed")
        raise HTTPException(status_code=500, detail=f"Gagal menghapus index: {exc}") from exc


@router.post("/ingest", response_model=IngestResponse)
def ingest(
    force: bool = Query(False),
    source_file: list[str] | None = Query(None),
) -> IngestResponse:
    try:
        result = ingest_job_manager.start(force=force, source_files=source_file)
        return IngestResponse(**result)
    except Exception as exc:
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=500, detail=f"Ingestion gagal: {exc}") from exc


@router.get("/ingest/status", response_model=IngestResponse)
def ingest_status() -> IngestResponse:
    try:
        return IngestResponse(**ingest_job_manager.get_status())
    except Exception as exc:
        logger.exception("Ingestion status failed")
        raise HTTPException(status_code=500, detail=f"Gagal mengambil status ingest: {exc}") from exc
