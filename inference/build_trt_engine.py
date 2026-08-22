from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent.parent
TORCH_DIR = ROOT_DIR / "inference" / "models" / "torch"
ONNX_DIR = ROOT_DIR / "inference" / "models" / "onnx"
ENGINE_DIR = ROOT_DIR / "inference" / "models" / "engine"


# builds .engine models, only int8 and fp16 precision are implemented
def build_engine(model_name: str, precision: str, img_size: int = 640) -> Path:
    ENGINE_DIR.mkdir(parents=True, exist_ok=True)
    pytorch_path = TORCH_DIR / f"{model_name}.pt"
    onnx_path = ONNX_DIR / f"{model_name}.onnx"

    # FP16
    if precision == "fp16":
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found: {onnx_path}\n"
                "First run download_models.py and export_onnx.py"
            )
        import tensorrt as trt

        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, logger)
        with open(onnx_path, "rb") as file:
            if not parser.parse(file.read()):
                for i in range(parser.num_errors):
                    print(parser.get_error(i))
                raise RuntimeError("ONNX parse failed")
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
        config.set_flag(trt.BuilderFlag.FP16)
        print(f"Building FP16 engine: {model_name}")
        engine_bytes = builder.build_serialized_network(network, config)
        if engine_bytes is None:
            raise RuntimeError("TensorRT engine build failed")
        output_path = ENGINE_DIR / f"{model_name}_fp16.engine"
        with open(output_path, "wb") as file:
            file.write(engine_bytes)
        print(f"Saved: {output_path}")
        return output_path

    # INT8 (COCO quantized)
    if precision == "int8":
        if not pytorch_path.exists():
            raise FileNotFoundError(f"Pytorch model not found: {pytorch_path}")
        print(f"Building INT8 engine: {model_name}")
        print("Using coco8.yaml for INT8 calibration")
        model = YOLO(pytorch_path)
        exported_path = model.export(
            format="engine",
            quantize=8,
            data="coco8.yaml",
            imgsz=img_size,
            batch=1,
            device=0,  # The jetson orin nano gpu
        )
        exported_path = Path(exported_path)
        output_path = ENGINE_DIR / f"{model_name}_int8.engine"

        # Ultralytics prepends a length prefixed JSON metadata header before
        # the raw TensorRT plan. Strip it so the file is a pure .engine that
        # trtexec / the C++ TensorRT runtime can load directly.
        import struct
        with open(exported_path, "rb") as f:
            data = f.read()
        meta_len = struct.unpack("<I", data[:4])[0]
        plan = data[4 + meta_len:]
        with open(output_path, "wb") as f:
            f.write(plan)
        if exported_path.resolve() != output_path.resolve():
            exported_path.unlink()
        print(f"Stripped {4 + meta_len} bytes of Ultralytics metadata")
        print(f"Saved raw TensorRT engine: {output_path}")
        return output_path

    raise ValueError(f"Unknown precision: {precision}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build TensorRT FP16 or INT8 engines")
    parser.add_argument(
        "model",
        help="Yolo models to download, yolo26n etc.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Build an FP16 TensorRT engine from ONNX",
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help="Build an INT8 TensorRT engine using coco8.yaml",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size (default: 640)",
    )
    args = parser.parse_args()
    if args.fp16 == args.int8:
        parser.error("Choose exactly one of --fp16 or --int8")
    precision = "fp16" if args.fp16 else "int8"
    build_engine(
        model_name=args.model,
        precision=precision,
        img_size=args.imgsz,
    )


if __name__ == "__main__":
    main()
