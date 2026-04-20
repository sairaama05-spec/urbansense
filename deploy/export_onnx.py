"""
UrbanSense – ONNX Export + Latency Benchmark
=============================================
Exports the best YOLOv8 checkpoint from training to ONNX (opset 14)
and runs a 50-iteration latency comparison between PyTorch and ONNX inference.

Usage
-----
    # from project root
    python deploy/export_onnx.py

Output
------
    deploy/urbansense_yolo.onnx
    Latency table printed to stdout
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── paths ─────────────────────────────────────────────────────────────────────
BEST_PT   = ROOT / "experiments" / "yolo_baseline" / "weights" / "best.pt"
ONNX_OUT  = ROOT / "deploy" / "urbansense_yolo.onnx"
IMGSZ     = 640
OPSET     = 14
N_ITERS   = 50
DUMMY_IMG = np.random.randint(0, 255, (IMGSZ, IMGSZ, 3), dtype=np.uint8)

# ── helpers ────────────────────────────────────────────────────────────────────

def _mean_ms(times_sec: list[float]) -> float:
    return round(float(np.mean(times_sec)) * 1000, 2)


def export_yolo_onnx(pt_path: Path, out_path: Path) -> None:
    """Export YOLOv8 .pt → ONNX using Ultralytics built-in exporter."""
    from ultralytics import YOLO

    print(f"Loading  : {pt_path}")
    model = YOLO(str(pt_path))

    print(f"Exporting: {out_path}  (opset={OPSET}, imgsz={IMGSZ})")
    model.export(
        format="onnx",
        imgsz=IMGSZ,
        opset=OPSET,
        simplify=True,
        dynamic=False,
    )

    # Ultralytics saves next to the .pt file; move to desired location
    default_onnx = pt_path.with_suffix(".onnx")
    if default_onnx.exists() and default_onnx != out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        default_onnx.rename(out_path)

    print(f"Saved    : {out_path}\n")


def benchmark_pytorch(pt_path: Path, n: int = N_ITERS) -> float:
    """Warm-up 3 runs then time n iterations; return avg latency in seconds."""
    from ultralytics import YOLO
    from PIL import Image

    model = YOLO(str(pt_path))
    img   = Image.fromarray(DUMMY_IMG)

    # warm-up
    for _ in range(3):
        model(img, verbose=False)

    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        model(img, verbose=False)
        times.append(time.perf_counter() - t0)

    return _mean_ms(times)


def benchmark_onnx(onnx_path: Path, n: int = N_ITERS) -> float:
    """Warm-up 3 runs then time n iterations using ONNX Runtime."""
    import onnxruntime as ort

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers()
        else ["CPUExecutionProvider"]
    )
    sess = ort.InferenceSession(str(onnx_path), providers=providers)
    inp_name = sess.get_inputs()[0].name

    # ONNX Runtime expects NCHW float32 normalised [0, 1]
    dummy = (DUMMY_IMG.transpose(2, 0, 1).astype(np.float32) / 255.0)[np.newaxis]

    # warm-up
    for _ in range(3):
        sess.run(None, {inp_name: dummy})

    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        sess.run(None, {inp_name: dummy})
        times.append(time.perf_counter() - t0)

    return _mean_ms(times)


def print_table(results: dict[str, float]) -> None:
    """Pretty-print latency results."""
    print("\n" + "=" * 44)
    print(f"  {'Backend':<18}  {'Avg latency (ms)':>16}")
    print("-" * 44)
    for name, ms in results.items():
        print(f"  {name:<18}  {ms:>16.2f}")
    print("=" * 44)
    if "PyTorch" in results and "ONNX" in results:
        speedup = results["PyTorch"] / results["ONNX"]
        print(f"  ONNX speedup: {speedup:.2f}x faster than PyTorch")
    print()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if not BEST_PT.exists():
        print(f"[WARNING] Trained weights not found at {BEST_PT}")
        print("  Falling back to pretrained YOLOv8n for benchmarking.\n")
        fallback = ROOT / "yolov8n.pt"
        pt_path  = fallback
    else:
        pt_path = BEST_PT

    # ── export ────────────────────────────────────────────────────────────────
    export_yolo_onnx(pt_path, ONNX_OUT)

    # ── benchmark ─────────────────────────────────────────────────────────────
    results: dict[str, float] = {}

    print(f"Benchmarking PyTorch ({N_ITERS} iters) …")
    results["PyTorch"] = benchmark_pytorch(pt_path)

    if ONNX_OUT.exists():
        print(f"Benchmarking ONNX   ({N_ITERS} iters) …")
        results["ONNX"] = benchmark_onnx(ONNX_OUT)
    else:
        print(f"[SKIP] ONNX file not found at {ONNX_OUT}")

    print_table(results)


if __name__ == "__main__":
    main()
