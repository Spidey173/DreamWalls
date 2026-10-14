from typing import Literal, Optional, List
from pydantic_settings import BaseSettings, SettingsConfigDict
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    PROJECT_NAME: str = "DreamWalls"
    VERSION: str = "1.0.0"

    # LLM Provider: 'gemini' | 'anthropic' | 'groq'
    LLM_PROVIDER: Literal["gemini", "anthropic", "groq"] = "gemini"

    # Gemini
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.5-flash"

    # Anthropic
    ANTHROPIC_API_KEY: str = ""
    ANTHROPIC_MODEL: str = "claude-3-5-sonnet-20241022"

    # Groq
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "openai/gpt-oss-120b"
    GROQ_VISION_MODEL: str = "qwen/qwen3.8-27b"

    # Database
    DATABASE_URL: str = "sqlite:///./paintmywall.db"

    # Security & CORS
    ADMIN_PASSWORD: str = "paintshop_admin_secret_2026"
    CORS_ORIGINS: str = "*"  # Comma-separated list of allowed origins, or "*"

    def get_cors_origins(self) -> List[str]:
        if not self.CORS_ORIGINS or self.CORS_ORIGINS.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    # Business configuration
    SHOP_WHATSAPP_NUMBER: str = "+919876543210"
    RATE_LIMIT_PER_MINUTE: int = 5

    # Image Editing Model Configuration
    IMAGE_MODEL: str = "gemini-2.5-flash-image"
    IMAGE_API_KEY: str = ""
    IMAGE_ALLOW_MOCK: bool = False
    IMAGE_GLOBAL_DAILY_CAP: int = 100
    IMAGE_PER_IP_DAILY_LIMIT: int = 10
    IMAGE_TIMEOUT_SECONDS: float = 35.0
    IMAGE_CACHE_DIR: Path = BASE_DIR / "uploads" / "image_cache"

    # Storage paths
    DATA_DIR: Path = BASE_DIR / "data"
    MODEL_DIR: Path = BASE_DIR / "models"
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    CACHE_DIR: Path = BASE_DIR / "uploads" / "masks"

    # ONNX Model
    SEGFORMER_MODEL_VARIANT: Literal["b0", "b2"] = "b2"
    SEGFORMER_MODEL_PATH: Optional[Path] = None

    def get_segformer_model_path(self) -> Path:
        if self.SEGFORMER_MODEL_PATH is not None and Path(self.SEGFORMER_MODEL_PATH).exists():
            return Path(self.SEGFORMER_MODEL_PATH)
        variant_path = BASE_DIR / "models" / f"segformer_{self.SEGFORMER_MODEL_VARIANT}_ade20k.onnx"
        if variant_path.exists():
            return variant_path
        b2_path = BASE_DIR / "models" / "segformer_b2_ade20k.onnx"
        if b2_path.exists():
            return b2_path
        b0_path = BASE_DIR / "models" / "segformer_b0_ade20k.onnx"
        if b0_path.exists():
            return b0_path
        return variant_path

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR.parent / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def model_post_init(self, __context):
        # Resolve MODEL_DIR if relative or offset
        if not (self.MODEL_DIR / "segformer_b2_ade20k.onnx").exists():
            if (BASE_DIR / "models" / "segformer_b2_ade20k.onnx").exists():
                self.MODEL_DIR = BASE_DIR / "models"
        # Resolve UPLOAD_DIR
        if str(self.UPLOAD_DIR).endswith("backend/uploads") and (BASE_DIR / "uploads").exists():
            self.UPLOAD_DIR = BASE_DIR / "uploads"
        self.CACHE_DIR = self.UPLOAD_DIR / "masks"
        self.IMAGE_CACHE_DIR = self.UPLOAD_DIR / "image_cache"

settings = Settings()

# Ensure runtime directories exist
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.CACHE_DIR.mkdir(parents=True, exist_ok=True)
settings.MODEL_DIR.mkdir(parents=True, exist_ok=True)
settings.IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
