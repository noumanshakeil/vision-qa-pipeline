"""CV demo entry for vision-qa-pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

def make_demo_image(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (320, 240), (30, 40, 60))
    d = ImageDraw.Draw(img)
    d.rectangle([40, 40, 200, 180], outline=(0, 200, 120), width=3)
    d.text((50, 200), "demo", fill=(240, 240, 240))
    img.save(path)
    return path

def analyze(path: Path) -> str:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    mean = arr.mean(axis=(0, 1))
    fingerprint = np.round(mean, 3).tolist()
    edges = np.abs(np.diff(arr.mean(axis=2), axis=0)).mean()
    return (
        f"image={path.name} size={img.size} mean_rgb={fingerprint} "
        f"edge_energy={edges:.4f}\n"
        "Hook: replace analyze() with CLIP/YOLO/MediaPipe for real CV."
    )

def main() -> None:
    p = argparse.ArgumentParser(description="Visual question answering pipeline: describe image context and answer user questions.")
    p.add_argument("--image", type=Path, default=Path("data/demo.png"))
    args = p.parse_args()
    if not args.image.exists():
        make_demo_image(args.image)
    print(analyze(args.image))

if __name__ == "__main__":
    main()
