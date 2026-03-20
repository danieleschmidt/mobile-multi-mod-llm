"""INT2Quantizer: simulate 2-bit weight quantization."""

import numpy as np


# The 4 quantization levels for 2-bit: {-1.5, -0.5, 0.5, 1.5}
_INT2_LEVELS = np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float32)


class INT2Quantizer:
    """
    Simulate 2-bit quantization via rounding to 4 levels.
    Weights are scaled to [-2, 2] range and quantized to {-1.5, -0.5, 0.5, 1.5}.
    """

    def __init__(self, bits: int = 2):
        self.bits = bits
        self.n_levels = 2 ** bits  # 4 for 2-bit

    def quantize(self, weights: np.ndarray) -> np.ndarray:
        """
        Quantize weights to 4 levels: {-1.5, -0.5, 0.5, 1.5}.

        Steps:
        1. Scale weights to [-2, 2] range
        2. Round to nearest of the 4 levels

        Args:
            weights: arbitrary shape float array

        Returns:
            quantized weights, same shape, values in {-1.5, -0.5, 0.5, 1.5}
        """
        w = weights.astype(np.float32)
        abs_max = np.abs(w).max()

        if abs_max == 0:
            return np.zeros_like(w)

        # Scale to [-2, 2]
        w_scaled = w * (2.0 / abs_max)
        w_clipped = np.clip(w_scaled, -2.0, 2.0)

        # Find nearest level for each element
        # levels: {-1.5, -0.5, 0.5, 1.5}
        # Boundaries: -1.0, 0.0, 1.0
        quantized = np.select(
            [w_clipped < -1.0, w_clipped < 0.0, w_clipped < 1.0],
            [-1.5, -0.5, 0.5],
            default=1.5,
        ).astype(np.float32)

        return quantized

    def dequantize(self, quantized: np.ndarray, scale: float, zero_point: float = 0.0) -> np.ndarray:
        """
        Dequantize weights back to float.

        Args:
            quantized: quantized weights (values in {-1.5, -0.5, 0.5, 1.5})
            scale: scale factor used during quantization (abs_max / 2)
            zero_point: zero point offset (default 0)

        Returns:
            float weights (approximate reconstruction)
        """
        return (quantized.astype(np.float32) - zero_point) * scale

    def quantize_model_params(self, params_dict: dict) -> tuple:
        """
        Quantize all arrays in a params dict.

        Args:
            params_dict: dict of name -> numpy array

        Returns:
            (quantized_dict, scales_dict)
            - quantized_dict: same keys, quantized values
            - scales_dict: same keys, scale factors for dequantization
        """
        quantized_dict = {}
        scales_dict = {}

        for name, weights in params_dict.items():
            w = weights.astype(np.float32)
            abs_max = np.abs(w).max()

            if abs_max == 0:
                quantized_dict[name] = np.zeros_like(w)
                scales_dict[name] = 1.0
            else:
                scale = abs_max / 2.0
                quantized_dict[name] = self.quantize(w)
                scales_dict[name] = float(scale)

        return quantized_dict, scales_dict

    def memory_footprint_bytes(self, params_dict: dict) -> int:
        """
        Calculate memory in bytes for INT2 storage.

        Args:
            params_dict: dict of name -> numpy array

        Returns:
            total bytes at `bits` bits per parameter
        """
        total_params = sum(arr.size for arr in params_dict.values())
        # bits / 8 bytes per param
        return int(total_params * self.bits // 8)
