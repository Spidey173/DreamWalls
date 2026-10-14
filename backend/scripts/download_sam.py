#!/usr/bin/env python3
"""
Downloads lightweight MobileSAM ONNX models (~26 MB total) for interactive tap-to-select segmentation.
"""

import os
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SAM_MODELS = [
    (
        "https://huggingface.co/aletrn/sam-quantized/resolve/main/machine_learning_models/mobile_sam.decoder.onnx",
        MODEL_DIR / "mobile_sam_decoder.onnx"
    ),
    (
        "https://huggingface.co/aletrn/sam-quantized/resolve/main/machine_learning_models/mobile_sam.encoder.quant.onnx",
        MODEL_DIR / "mobile_sam_encoder.onnx"
    )
]

def download_models():
    for url, target_path in SAM_MODELS:
        if target_path.exists() and target_path.stat().st_size > 5_000_000:
            print(f"✓ {target_path.name} already exists ({target_path.stat().st_size / 1024 / 1024:.1f} MB). Skipping.")
            continue

        print(f"Downloading {target_path.name} from Hugging Face...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp, open(target_path, "wb") as out_file:
            out_file.write(resp.read())
        print(f"✓ Successfully saved {target_path.name} ({target_path.stat().st_size / 1024 / 1024:.1f} MB).")

if __name__ == "__main__":
    download_models()
