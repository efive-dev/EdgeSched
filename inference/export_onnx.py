from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent.parent
TORCH_DIR = ROOT_DIR / "inference" / "models" / "torch"
ONNX_DIR = ROOT_DIR / "inference" / "models" / "onnx"


def export_onnx(model_name: str, img_size: int = 640) -> Path:
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    pytorch_path = TORCH_DIR / f"{model_name}.pt"
    if not pytorch_path.exists():
        raise FileNotFoundError(
            f"Pytorch model not found: {pytorch_path}\n"
            f"First run download_yolo_models.py"
        )
    print(f"Loading: {pytorch_path}")
    model = YOLO(pytorch_path)
    print("Exporting from pytorch to ONNX")
    exported_path = model.export(
        format="onnx",
        imgsz=img_size,
        device="cpu",  # or whatever you have available
    )
    exported_path = Path(exported_path)
    output_path = ONNX_DIR / f"{model_name}.onnx"
    if exported_path.resolve() != output_path.resolve():
        shutil.move(str(exported_path), str(output_path))
    print(f"Saved: {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "models", nargs="+", help="Yolo models to download, yolo26n etc."
    )
    parser.add_argument("--imgsize", type=int, default=640, help="Input image size")
    args = parser.parse_args()
    for model_name in args.models:
        export_onnx(model_name, args.imgsize)


if __name__ == "__main__":
    main()
