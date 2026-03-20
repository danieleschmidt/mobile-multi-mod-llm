"""Benchmark utilities for mobile multi-modal LLM."""

import numpy as np

from .vision_encoder import TinyVisionEncoder
from .text_decoder import TinyTextDecoder
from .fusion import CrossModalFusion
from .quantizer import INT2Quantizer


def param_count(params_dict: dict) -> int:
    """Sum of all array sizes in a params dict."""
    return sum(arr.size for arr in params_dict.values())


def benchmark_model() -> dict:
    """
    Instantiate all model components and compute stats.

    Returns:
        dict with keys:
          - vision_params: int
          - text_params: int
          - fusion_params: int
          - total_params: int
          - memory_mb_fp32: float
          - memory_mb_int2: float
    """
    vision = TinyVisionEncoder()
    text = TinyTextDecoder()
    fusion = CrossModalFusion()
    quantizer = INT2Quantizer(bits=2)

    vision_params = vision.param_count()
    text_params = text.param_count()
    fusion_params = fusion.param_count()
    total_params = vision_params + text_params + fusion_params

    # Memory in MB for float32 (4 bytes per param)
    memory_mb_fp32 = total_params * 4 / (1024 ** 2)

    # Collect all params for INT2 memory estimate
    all_params = {}
    for k, v in vision.params.items():
        all_params[f"vision_{k}"] = v
    for k, v in text.params.items():
        all_params[f"text_{k}"] = v
    for k, v in fusion.params.items():
        all_params[f"fusion_{k}"] = v

    memory_footprint_bytes = quantizer.memory_footprint_bytes(all_params)
    memory_mb_int2 = memory_footprint_bytes / (1024 ** 2)

    return {
        "vision_params": vision_params,
        "text_params": text_params,
        "fusion_params": fusion_params,
        "total_params": total_params,
        "memory_mb_fp32": memory_mb_fp32,
        "memory_mb_int2": memory_mb_int2,
    }
