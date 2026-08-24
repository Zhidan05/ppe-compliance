import os
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import psutil
import torch
from ultralytics import YOLO

# ============================================================
# PATH PROJECT & MODEL REGISTRY
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

PYTORCH_MODEL = MODELS_DIR / "pytorch" / "best.pt"
ONNX_MODEL = MODELS_DIR / "onnx" / "best.onnx"
# Arahkan langsung ke file XML untuk OpenVINO
OPENVINO_MODEL = MODELS_DIR / "openvino" / "best.xml"
TENSORRT_MODEL = MODELS_DIR / "tensorrt" / "best.engine"

# ============================================================
# BENCHMARK CONFIGURATION
# ============================================================
WARMUP_RUNS = 10
BENCHMARK_RUNS = 100
IMAGE_SIZES = [640]

try:
    ort.preload_dlls()
except Exception:
    pass


# ============================================================
# SYSTEM PROFILING & HEALTH CHECKS
# ============================================================
def get_system_info() -> dict[str, Any]:
    has_cuda = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if has_cuda else "Tidak Terdeteksi (CPU)"

    return {
        "os": f"{platform.system()} {platform.release()}",
        "cpu": platform.processor(),
        "cores": f"{psutil.cpu_count(logical=False)} Physical / {psutil.cpu_count(logical=True)} Logical",
        "ram": f"{psutil.virtual_memory().total / (1024 ** 3):.2f} GB",
        "gpu": gpu_name,
        "cuda_available": has_cuda,
    }


def check_onnx_cuda() -> bool:
    try:
        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" not in providers:
            return False

        if not ONNX_MODEL.exists():
            return False

        session = ort.InferenceSession(
            str(ONNX_MODEL),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        return "CUDAExecutionProvider" in session.get_providers()
    except Exception as e:
        print(f"\n[WARNING] ONNX CUDA provider check gagal: {e}")
        return False


# ============================================================
# CORE BENCHMARK ENGINE
# ============================================================
def run_benchmark(
    model_path: Path,
    device: int | str,
    imgsz: int,
) -> dict[str, Any]:
    try:
        # Gunakan string path yang dinormalisasi
        model = YOLO(str(model_path), task="detect")
    except Exception as e:
        return {"error": f"Gagal memuat model: {e}"}

    dummy_frame = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)

    try:
        # 1. Warmup Loop
        for _ in range(WARMUP_RUNS):
            _ = model.predict(
                source=dummy_frame,
                imgsz=imgsz,
                device=device,
                verbose=False,
            )

        # 2. Benchmark Loop
        latencies = []
        cpu_usages = []

        if device == 0 and torch.cuda.is_available():
            torch.cuda.synchronize()

        start_total_time = time.perf_counter()

        for _ in range(BENCHMARK_RUNS):
            cpu_usages.append(psutil.cpu_percent(interval=None))
            t0 = time.perf_counter()

            _ = model.predict(
                source=dummy_frame,
                imgsz=imgsz,
                device=device,
                verbose=False,
            )

            if device == 0 and torch.cuda.is_available():
                torch.cuda.synchronize()

            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)

        total_time = time.perf_counter() - start_total_time
        avg_latency = np.mean(latencies)
        fps = BENCHMARK_RUNS / total_time
        avg_cpu = np.mean(cpu_usages)

        process = psutil.Process(os.getpid())
        ram_used_mb = process.memory_info().rss / (1024 * 1024)

        gpu_mem_mb = 0.0
        if device == 0 and torch.cuda.is_available():
            gpu_mem_mb = torch.cuda.memory_allocated(0) / (1024 * 1024)

        return {
            "avg_latency_ms": float(avg_latency),
            "p95_latency_ms": float(np.percentile(latencies, 95)),
            "fps": float(fps),
            "avg_cpu_percent": float(avg_cpu),
            "ram_mb": float(ram_used_mb),
            "gpu_mem_mb": float(gpu_mem_mb),
            "error": None,
        }

    except Exception as e:
        return {"error": str(e)}


# ============================================================
# MAIN ENTRY POINT
# ============================================================
def main():
    sys_info = get_system_info()

    print("=" * 80)
    print(f"{'BENCHMARK PERFORMA MODEL PPE':^80}")
    print("=" * 80)
    print(f"OS           : {sys_info['os']}")
    print(f"CPU          : {sys_info['cpu']} ({sys_info['cores']})")
    print(f"Total RAM    : {sys_info['ram']}")
    print(f"GPU / CUDA   : {sys_info['gpu']}")
    print(f"Project Root : {BASE_DIR}")
    print("=" * 80)

    target_models = [
        ("TensorRT", TENSORRT_MODEL, 0),
        ("ONNX (CUDA)", ONNX_MODEL, 0),
        ("PyTorch (CUDA)", PYTORCH_MODEL, 0),
        ("OpenVINO (CPU)", OPENVINO_MODEL, "cpu"),
        ("ONNX (CPU)", ONNX_MODEL, "cpu"),
        ("PyTorch (CPU)", PYTORCH_MODEL, "cpu"),
    ]

    results: list[dict[str, Any]] = []

    print(f"\nMenjalankan benchmark ({BENCHMARK_RUNS} iterasi per pipeline)...\n")

    for label, model_path, device in target_models:
        # Validasi CUDA
        if device == 0 and not sys_info["cuda_available"]:
            print(f"-> {label:<18} [SKIPPED] CUDA tidak tersedia.")
            continue

        # Validasi Keberadaan Model File
        if not model_path.exists():
            print(f"-> {label:<18} [SKIPPED] File tidak ditemukan: {model_path}")
            continue

        # Validasi ONNX CUDA Provider
        if label == "ONNX (CUDA)" and not check_onnx_cuda():
            print(f"-> {label:<18} [SKIPPED] CUDAExecutionProvider tidak aktif.")
            continue

        for imgsz in IMAGE_SIZES:
            print(f"-> Testing: {label:<16} | Res: {imgsz}x{imgsz} | Device: {device} ...", end="", flush=True)

            res = run_benchmark(
                model_path=model_path,
                device=device,
                imgsz=imgsz,
            )

            if res.get("error"):
                print(f" [FAILED] ({res['error']})")
                continue

            print(f" [OK] -> {res['fps']:.1f} FPS ({res['avg_latency_ms']:.2f} ms)")

            res.update({
                "label": label,
                "file": model_path.name,
                "imgsz": imgsz,
                "device": str(device),
            })
            results.append(res)

    if not results:
        print("\n[PERINGATAN] Tidak ada benchmark yang berhasil dijalankan.")
        return

    results.sort(key=lambda x: x["fps"], reverse=True)

    print("\n" + "=" * 90)
    print(
        f"{'Format / Pipeline':<18} | "
        f"{'Res':<6} | "
        f"{'Avg Latency':<12} | "
        f"{'FPS':<8} | "
        f"{'CPU %':<7} | "
        f"{'RAM (MB)':<9} | "
        f"{'GPU MB':<8}"
    )
    print("-" * 90)

    for r in results:
        print(
            f"{r['label']:<18} | "
            f"{r['imgsz']:<6} | "
            f"{r['avg_latency_ms']:>7.2f} ms   | "
            f"{r['fps']:>6.1f} | "
            f"{r['avg_cpu_percent']:>5.1f}% | "
            f"{r['ram_mb']:>8.1f} | "
            f"{r['gpu_mem_mb']:>7.1f}"
        )

    print("=" * 90)

    best = results[0]
    print("\n[REKOMENDASI TERBAIK]")
    print(f"Model Pilihan : {best['label']} ({best['file']})")
    print(f"Throughput    : {best['fps']:.2f} FPS")
    print(f"Latensi Rata2 : {best['avg_latency_ms']:.2f} ms (p95: {best['p95_latency_ms']:.2f} ms)")


if __name__ == "__main__":
    main()