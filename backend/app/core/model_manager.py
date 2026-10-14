import ssl
import sys
import certifi
import urllib.request
import logging
from pathlib import Path
from typing import Dict, Tuple

from app.core.config import settings

logger = logging.getLogger("paintmywall.models")

SEGFORMER_URLS: Dict[str, str] = {
    "b0": "https://huggingface.co/Xenova/segformer-b0-finetuned-ade-512-512/resolve/main/onnx/model.onnx",
    "b2": "https://huggingface.co/Xenova/segformer-b2-finetuned-ade-512-512/resolve/main/onnx/model.onnx",
}

SAM_MODELS: Dict[str, str] = {
    "mobile_sam_decoder.onnx": "https://huggingface.co/aletrn/sam-quantized/resolve/main/machine_learning_models/mobile_sam.decoder.onnx",
    "mobile_sam_encoder.onnx": "https://huggingface.co/aletrn/sam-quantized/resolve/main/machine_learning_models/mobile_sam.encoder.quant.onnx",
}

def _download_file(url: str, target_path: Path, min_bytes: int = 5_000_000) -> bool:
    """Downloads a file if not present or incomplete, using secure SSL context."""
    if target_path.exists() and target_path.stat().st_size >= min_bytes:
        return False

    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_suffix(".tmp")
    logger.info(f"Downloading {target_path.name} from {url}...")

    ctx = ssl.create_default_context(cafile=certifi.where())
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "PaintMyWall/1.0"}
    )

    try:
        with urllib.request.urlopen(req, context=ctx) as response, open(temp_path, "wb") as out_file:
            while True:
                chunk = response.read(1024 * 64)
                if not chunk:
                    break
                out_file.write(chunk)

        if temp_path.stat().st_size < min_bytes:
            temp_path.unlink()
            raise ValueError(f"Downloaded file {temp_path.name} is smaller than expected min size ({min_bytes} bytes).")

        temp_path.replace(target_path)
        logger.info(f"Successfully downloaded {target_path.name} ({target_path.stat().st_size / (1024 * 1024):.1f} MB).")
        return True
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        logger.error(f"Failed to download {target_path.name} from {url}: {e}")
        raise

def ensure_models_downloaded():
    """
    Called on FastAPI application startup (lifespan).
    Ensures SegFormer (B2 default or B0) and MobileSAM ONNX models are present on disk.
    If missing, downloads them automatically and logs their sizes.
    """
    settings.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 1. SegFormer Model
    variant = settings.SEGFORMER_MODEL_VARIANT.lower()
    target_segformer = settings.MODEL_DIR / f"segformer_{variant}_ade20k.onnx"
    if variant in SEGFORMER_URLS:
        _download_file(SEGFORMER_URLS[variant], target_segformer, min_bytes=10_000_000)

    actual_segformer = settings.get_segformer_model_path()
    if actual_segformer.exists():
        size_mb = actual_segformer.stat().st_size / (1024 * 1024)
        logger.info(f"[Model Startup] ✓ SegFormer Model Loaded: {actual_segformer.name} ({size_mb:.1f} MB)")
    else:
        logger.warning(f"[Model Startup] ✗ SegFormer model not found at {actual_segformer}!")

    # 2. MobileSAM Models
    for filename, url in SAM_MODELS.items():
        sam_path = settings.MODEL_DIR / filename
        _download_file(url, sam_path, min_bytes=5_000_000)
        size_mb = sam_path.stat().st_size / (1024 * 1024)
        logger.info(f"[Model Startup] ✓ MobileSAM Model Loaded: {sam_path.name} ({size_mb:.1f} MB)")
