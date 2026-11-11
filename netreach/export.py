"""
CPU Optimization & INT8 Dynamic Quantization Export for NetReach.

Provides automated dynamic quantization (torch.ao.quantization.quantize_dynamic)
to optimize ChallengerLatentNetV4 for fast, low-memory CPU deployments
without requiring CUDA or NVIDIA hardware.
"""

import os
import time
from typing import Dict, Any, Optional, Union
import torch
import torch.nn as nn
import torch.ao.quantization as quantization


def quantize_model_for_cpu(
    model: nn.Module,
    dtype: torch.dtype = torch.qint8,
) -> nn.Module:
    """
    Quantize linear layers in ChallengerLatentNetV4 using INT8 dynamic quantization.

    Args:
        model: PyTorch model (ChallengerLatentNetV4 or baseline).
        dtype: Target quantized integer data type (default: torch.qint8).

    Returns:
        Quantized model optimized for CPU inference with ~4x reduced memory footprint.
    """
    model_cpu = model.to("cpu")
    model_cpu.eval()

    quantized = quantization.quantize_dynamic(
        model_cpu,
        {nn.Linear},
        dtype=dtype,
    )
    return quantized


def export_quantized_checkpoint(
    model: nn.Module,
    output_path: str,
    quantize: bool = True,
) -> str:
    """
    Export model weights to a quantized checkpoint on disk.

    Args:
        model: ChallengerLatentNetV4 model.
        output_path: Destination .pt file path.
        quantize: Whether to apply INT8 dynamic quantization before saving.

    Returns:
        str: Absolute path to the saved checkpoint.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    if quantize:
        export_model = quantize_model_for_cpu(model)
    else:
        export_model = model.to("cpu")

    torch.save(
        {
            "model_state_dict": export_model.state_dict(),
            "quantized": quantize,
            "architecture": "ChallengerLatentNetV4",
        },
        output_path,
    )
    return os.path.abspath(output_path)


def benchmark_cpu_latency(
    model: nn.Module,
    num_nodes: int = 256,
    target_edges: int = 768,
    steps: int = 2,
    num_runs: int = 10,
    warmup_runs: int = 2,
) -> Dict[str, float]:
    """
    Benchmark per-graph inference latency on CPU.
    """
    from data.generator import create_dataloader

    model.to("cpu").eval()
    loader = create_dataloader(
        hop_lengths=[6],
        num_samples=max(num_runs + warmup_runs, 4),
        batch_size=1,
        num_nodes=num_nodes,
        target_edges=target_edges,
        shuffle=False,
    )

    batches = [{k: v.to("cpu") for k, v in b.items()} for b in loader]

    # Warmup
    with torch.no_grad():
        for i in range(min(warmup_runs, len(batches))):
            model(batches[i], steps=steps)

    latencies = []
    with torch.no_grad():
        for i in range(warmup_runs, min(warmup_runs + num_runs, len(batches))):
            t0 = time.perf_counter()
            model(batches[i], steps=steps)
            latencies.append((time.perf_counter() - t0) * 1000.0)

    if not latencies:
        latencies = [0.0]

    return {
        "mean_ms": sum(latencies) / len(latencies),
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "runs": len(latencies),
    }
