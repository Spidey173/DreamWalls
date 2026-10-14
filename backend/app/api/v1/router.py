from fastapi import APIRouter
from app.api.v1.endpoints import health, analyze, recolor, admin

api_router = APIRouter()

api_router.include_router(health.router, tags=["Health"])
api_router.include_router(analyze.router, tags=["Analysis & Recolor"])
api_router.include_router(recolor.router, tags=["Mask Correction"])
api_router.include_router(admin.router, prefix="/admin", tags=["Admin Catalog"])
