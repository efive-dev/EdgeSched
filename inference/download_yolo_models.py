from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent.parent
TORCH_DIR = ROOT_DIR / "inference" / "models" / "torch"

def download_model(model_name: str) -> Path:
    TORCH_DIR.mkdir(parents=True, exist_ok=True)
    output_path = TORCH_DIR / f"{model_name}.pt"
    if output_path.exists():
        print(f"Model already exists: {output_path}")
        return output_path
    print(f"Downloading {model_name}.pt")
    model = YOLO(f"{model_name}.pt")
    source_path = Path(model.ckpt_path)
    source_path.replace(output_path)
    print(f"Saved: {output_path}")
    return output_path

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "models",
        nargs="+",
        help="Yolo models to download, yolo26n etc."
    )
    args = parser.parse_args()
    for model_name in args.models:
        download_model(model_name)

if __name__ == "__main__":
    main()
