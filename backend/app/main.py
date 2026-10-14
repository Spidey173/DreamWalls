import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.core.database import Base, engine
from app.api.v1.router import api_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("paintmywall")

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing database schema...")
    Base.metadata.create_all(bind=engine)

    # Ensure all required ONNX models are present on startup
    from app.core.model_manager import ensure_models_downloaded
    logger.info("Verifying ONNX model files...")
    ensure_models_downloaded()

    origins = settings.get_cors_origins()
    logger.info(f"Configured CORS origins: {origins}")
    logger.info("DreamWalls API initialized successfully and ready for traffic.")
    yield

def create_app() -> FastAPI:
    app = FastAPI(
        title="DreamWalls API",
        description="Asian Paints AI Architectural Recolor and Shade Suggestion Engine",
        version=settings.VERSION,
        lifespan=lifespan
    )

    # Enable configurable CORS for frontend interaction
    cors_origins = settings.get_cors_origins()
    allow_credentials = False if cors_origins == ["*"] else True
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount static directory for uploaded & repainted images
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    app.mount("/static/uploads", StaticFiles(directory=str(settings.UPLOAD_DIR)), name="uploads")

    # Include API routes
    app.include_router(api_router, prefix="/api/v1")
    app.include_router(api_router, prefix="/api")  # alias

    @app.get("/")
    def root():
        return {
            "app": settings.PROJECT_NAME,
            "version": settings.VERSION,
            "docs": "/docs",
            "health": "/api/v1/health"
        }

    return app

app = create_app()
