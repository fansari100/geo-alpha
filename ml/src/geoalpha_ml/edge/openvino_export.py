"""
OpenVINO INT8 export for edge inference.

Same trick I use to squeeze the trading model down to 4x faster
inference: post-training INT8 quantization with a small calibration
set.  Here the deployment target is on-board the satellite payload
(or a ground-station edge node) where the workload is similar -
sub-10 ms decisions on small batches with a tight power envelope.

The two-stage flow is:
    PyTorch  ->  ONNX        (export_to_onnx)
    ONNX     ->  OpenVINO IR  (offline tooling, see scripts/quantize.sh)
              ->  INT8 IR     (quantize_int8 below, NNCF API)

All three artifacts ship side-by-side so deployment can pick whichever
the target runtime supports.  The lat measurements at the bottom are
the contract the on-board scheduler depends on.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch


@dataclass
class OpenVINOExportConfig:
    output_dir: Path = Path("artifacts/edge")
    onnx_opset: int = 17
    sample_batch: int = 1
    seq_len: int = 32
    in_features: int = 8
    int8_calibration_samples: int = 256
    target_latency_ms: float = 5.0


def export_to_onnx(model: torch.nn.Module, cfg: OpenVINOExportConfig) -> Path:
    """Trace + export the model to ONNX."""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.output_dir / "model.onnx"
    model.eval()
    dummy = torch.randn(cfg.sample_batch, cfg.seq_len, cfg.in_features)
    torch.onnx.export(
        model,
        dummy,
        out.as_posix(),
        input_names=["x"],
        output_names=["mean", "logvar"],
        dynamic_axes={"x": {0: "batch"}, "mean": {0: "batch"}, "logvar": {0: "batch"}},
        opset_version=cfg.onnx_opset,
    )
    return out


def quantize_int8(
    onnx_path: Path,
    calibration_loader: Iterable[np.ndarray],
    cfg: OpenVINOExportConfig,
) -> Path:
    """Post-training INT8 quantization via NNCF + OpenVINO.

    NNCF is imported lazily so the rest of this module loads even on
    machines without OpenVINO installed (e.g. CI without the runtime).
    """
    try:
        import openvino as ov
        import nncf
    except ImportError as exc:
        raise RuntimeError(
            "OpenVINO + NNCF are required for INT8 quantization - "
            "pip install openvino nncf"
        ) from exc

    core = ov.Core()
    model = core.read_model(onnx_path.as_posix())

    def _calibration_dataset():
        for sample in calibration_loader:
            yield {"x": sample.astype(np.float32)}

    quantized = nncf.quantize(
        model,
        nncf.Dataset(list(_calibration_dataset())),
        preset=nncf.QuantizationPreset.MIXED,
    )
    out = cfg.output_dir / "model.int8.xml"
    ov.save_model(quantized, out.as_posix())
    return out


def benchmark_latency(
    onnx_or_xml_path: Path,
    sample: np.ndarray,
    n_iter: int = 256,
) -> dict:
    """Wallclock latency of one inference path."""
    import time

    try:
        import openvino as ov
    except ImportError as exc:
        raise RuntimeError("openvino is required for benchmarking") from exc

    core = ov.Core()
    compiled = core.compile_model(onnx_or_xml_path.as_posix(), "CPU")
    infer = compiled.create_infer_request()
    sample = sample.astype(np.float32)
    # Warm-up.
    for _ in range(8):
        infer.infer({0: sample})
    times = np.empty(n_iter)
    for i in range(n_iter):
        t0 = time.perf_counter()
        infer.infer({0: sample})
        times[i] = (time.perf_counter() - t0) * 1000.0
    return {
        "mean_ms": float(times.mean()),
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
        "p99_ms": float(np.percentile(times, 99)),
        "max_ms": float(times.max()),
        "n_iter": n_iter,
    }
