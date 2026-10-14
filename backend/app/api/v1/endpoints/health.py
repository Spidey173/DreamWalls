from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.core.database import get_db
from app.core.config import settings

router = APIRouter()

@router.get("/health")
def health_check(db: Session = Depends(get_db)):
    db_status = "healthy"
    try:
        db.execute(text("SELECT 1"))
    except Exception as e:
        db_status = f"unhealthy: {e}"

    model_path = settings.get_segformer_model_path()
    model_status = "ready" if model_path.exists() else "missing"

    import os
    import resource
    # On Linux ru_maxrss is in kilobytes; on macOS it is in bytes
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mb = round(rss_kb / 1024.0 if os.name != "darwin" else rss_kb / (1024.0 * 1024.0), 2)

    return {
        "status": "ok",
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "database": db_status,
        "segmentation_model": model_status,
        "model_file": model_path.name,
        "model_variant": settings.SEGFORMER_MODEL_VARIANT,
        "llm_provider": settings.LLM_PROVIDER,
        "memory_rss_mb": rss_mb
    }
