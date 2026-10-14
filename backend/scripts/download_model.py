import os
import ssl
import sys
import certifi
import urllib.request
import argparse
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

MODEL_URLS = {
    "b0": "https://huggingface.co/Xenova/segformer-b0-finetuned-ade-512-512/resolve/main/onnx/model.onnx",
    "b2": "https://huggingface.co/Xenova/segformer-b2-finetuned-ade-512-512/resolve/main/onnx/model.onnx",
}

def download_model(variant: str = "b2", force: bool = False) -> Path:
    variant = variant.lower()
    if variant not in MODEL_URLS:
        raise ValueError(f"Unknown model variant '{variant}'. Choose 'b0' or 'b2'.")

    target_path = MODEL_DIR / f"segformer_{variant}_ade20k.onnx"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if target_path.exists() and not force:
        print(f"SegFormer {variant.upper()} ONNX model already exists at: {target_path}")
        return target_path

    url = MODEL_URLS[variant]
    print(f"Downloading SegFormer {variant.upper()} (ADE20K) ONNX model to {target_path}...")
    ctx = ssl.create_default_context(cafile=certifi.where())
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "PaintMyWall/1.0"}
    )

    try:
        with urllib.request.urlopen(req, context=ctx) as response, open(target_path, "wb") as out_file:
            total_size = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 1024 * 64

            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    sys.stdout.write(f"\rProgress: {downloaded}/{total_size} bytes ({percent:.1f}%)")
                    sys.stdout.flush()

        print(f"\nModel successfully saved to {target_path} ({target_path.stat().st_size} bytes)")
        return target_path
    except Exception as e:
        if target_path.exists():
            target_path.unlink()
        print(f"\nFailed to download model from {url}: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download SegFormer ADE20K ONNX model")
    parser.add_argument("--variant", choices=["b0", "b2"], default="b2", help="Model variant (b0 or b2)")
    parser.add_argument("--force", action="store_true", help="Force re-download even if exists")
    args = parser.parse_args()
    download_model(variant=args.variant, force=args.force)
