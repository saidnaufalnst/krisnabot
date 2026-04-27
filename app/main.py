from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.admin import router as admin_router
from app.api.health import router as health_router
from app.api.user import router as user_router
from app.core.config import settings
from app.db.session import init_db
from app.services.ingestion_service import get_ingestion_service

app = FastAPI(title=settings.app_name)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_origin_regex=settings.cors_origin_regex or None,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(health_router)
app.include_router(admin_router)
app.include_router(user_router)


@app.on_event("startup")
def on_startup() -> None:
    init_db()
    get_ingestion_service().run_startup_cleanup()
