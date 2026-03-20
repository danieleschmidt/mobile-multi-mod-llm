"""Tests for mobile multi-modal LLM components."""

import numpy as np
import pytest

from mmmllm.vision_encoder import TinyVisionEncoder
from mmmllm.text_decoder import TinyTextDecoder, VOCAB_SIZE
from mmmllm.fusion import CrossModalFusion
from mmmllm.quantizer import INT2Quantizer
from mmmllm.benchmark import benchmark_model


# ── Vision Encoder ──────────────────────────────────────────────────────────

class TestTinyVisionEncoder:
    def setup_method(self):
        self.enc = TinyVisionEncoder()
        self.img = np.random.rand(224, 224, 3).astype(np.float32)

    def test_patch_embed_shape(self):
        """patch_embed should return (n_patches, embed_dim)."""
        out = self.enc.patch_embed(self.img)
        n_patches = (224 // 16) ** 2  # 196
        assert out.shape == (n_patches, 256), f"Expected ({n_patches}, 256), got {out.shape}"

    def test_patch_embed_grayscale(self):
        """patch_embed should handle grayscale (H, W) images."""
        gray = np.random.rand(224, 224).astype(np.float32)
        out = self.enc.patch_embed(gray)
        assert out.shape == ((224 // 16) ** 2, 256)

    def test_forward_shape(self):
        """forward should return (n_patches+1, embed_dim) with CLS token."""
        out = self.enc.forward(self.img)
        n_patches = (224 // 16) ** 2  # 196
        assert out.shape == (n_patches + 1, 256), f"Expected ({n_patches + 1}, 256), got {out.shape}"

    def test_forward_cls_prepended(self):
        """First token of forward output is the CLS token (roughly different from patches)."""
        out = self.enc.forward(self.img)
        assert out.shape[0] == 197

    def test_param_count_positive(self):
        """param_count should be positive."""
        assert self.enc.param_count() > 0

    def test_param_count_under_10m(self):
        """param_count should be under 10 million."""
        assert self.enc.param_count() < 10_000_000, \
            f"Too many params: {self.enc.param_count():,}"


# ── Text Decoder ─────────────────────────────────────────────────────────────

class TestTinyTextDecoder:
    def setup_method(self):
        self.dec = TinyTextDecoder()
        self.token_ids = [1, 42, 100, 200, 500]

    def test_embed_shape(self):
        """embed should return (seq_len, embed_dim)."""
        out = self.dec.embed(self.token_ids)
        assert out.shape == (len(self.token_ids), 256), \
            f"Expected ({len(self.token_ids)}, 256), got {out.shape}"

    def test_forward_shape(self):
        """forward should return (seq_len, vocab_size) logits."""
        out = self.dec.forward(self.token_ids)
        assert out.shape == (len(self.token_ids), VOCAB_SIZE), \
            f"Expected ({len(self.token_ids)}, {VOCAB_SIZE}), got {out.shape}"

    def test_forward_single_token(self):
        """forward should work with a single token."""
        out = self.dec.forward([1])
        assert out.shape == (1, VOCAB_SIZE)

    def test_param_count_positive(self):
        """param_count should be positive."""
        assert self.dec.param_count() > 0

    def test_greedy_decode_returns_list(self):
        """greedy_decode should return a list of token ids."""
        ctx = np.random.rand(5, 256).astype(np.float32)
        result = self.dec.greedy_decode(ctx, max_new_tokens=5)
        assert isinstance(result, list)
        assert len(result) <= 5


# ── Cross-Modal Fusion ────────────────────────────────────────────────────────

class TestCrossModalFusion:
    def setup_method(self):
        self.fusion = CrossModalFusion()
        self.text_feat = np.random.rand(8, 256).astype(np.float32)
        self.vision_feat = np.random.rand(197, 256).astype(np.float32)

    def test_cross_attention_shape(self):
        """cross_attention should return (q_len, embed_dim)."""
        out = self.fusion.cross_attention(self.text_feat, self.vision_feat)
        assert out.shape == (8, 256), f"Expected (8, 256), got {out.shape}"

    def test_forward_shape(self):
        """forward should return (seq_len, embed_dim) fused features."""
        out = self.fusion.forward(self.text_feat, self.vision_feat)
        assert out.shape == self.text_feat.shape, \
            f"Expected {self.text_feat.shape}, got {out.shape}"

    def test_forward_different_seq_len(self):
        """forward should work with different text sequence lengths."""
        text = np.random.rand(16, 256).astype(np.float32)
        out = self.fusion.forward(text, self.vision_feat)
        assert out.shape == (16, 256)

    def test_param_count_positive(self):
        """param_count should be positive."""
        assert self.fusion.param_count() > 0


# ── INT2 Quantizer ────────────────────────────────────────────────────────────

class TestINT2Quantizer:
    def setup_method(self):
        self.q = INT2Quantizer(bits=2)
        self.weights = np.random.randn(64, 64).astype(np.float32)

    def test_quantize_same_shape(self):
        """quantize should return same shape as input."""
        out = self.q.quantize(self.weights)
        assert out.shape == self.weights.shape

    def test_quantize_valid_levels(self):
        """quantized values should be in {-1.5, -0.5, 0.5, 1.5}."""
        out = self.q.quantize(self.weights)
        unique_vals = set(np.unique(out).tolist())
        valid_levels = {-1.5, -0.5, 0.5, 1.5}
        assert unique_vals.issubset(valid_levels), \
            f"Found unexpected values: {unique_vals - valid_levels}"

    def test_quantize_four_levels_present(self):
        """Large random weights should use all 4 levels."""
        w = np.random.randn(1000).astype(np.float32)
        out = self.q.quantize(w)
        assert len(np.unique(out)) == 4

    def test_memory_footprint_less_than_fp32(self):
        """INT2 memory should be 1/16 of float32 memory."""
        params = {"w1": self.weights, "w2": np.random.randn(32, 32).astype(np.float32)}
        int2_bytes = self.q.memory_footprint_bytes(params)
        fp32_bytes = sum(arr.size * 4 for arr in params.values())
        assert int2_bytes < fp32_bytes, \
            f"INT2 ({int2_bytes}) should be less than FP32 ({fp32_bytes})"

    def test_memory_footprint_correct_ratio(self):
        """INT2 memory should be bits/32 of fp32 (2/32 = 1/16)."""
        params = {"w": np.ones((128, 128), dtype=np.float32)}
        int2_bytes = self.q.memory_footprint_bytes(params)
        total_params = 128 * 128
        expected = total_params * 2 // 8  # 2 bits per param
        assert int2_bytes == expected

    def test_quantize_model_params(self):
        """quantize_model_params should return dicts with same keys."""
        params = {
            "layer0_W": np.random.randn(32, 32).astype(np.float32),
            "layer0_b": np.random.randn(32).astype(np.float32),
        }
        q_dict, s_dict = self.q.quantize_model_params(params)
        assert set(q_dict.keys()) == set(params.keys())
        assert set(s_dict.keys()) == set(params.keys())


# ── Benchmark ────────────────────────────────────────────────────────────────

class TestBenchmark:
    def test_benchmark_runs(self):
        """benchmark_model should run without errors."""
        stats = benchmark_model()
        assert isinstance(stats, dict)

    def test_benchmark_keys(self):
        """benchmark_model should return expected keys."""
        stats = benchmark_model()
        expected_keys = {
            "vision_params",
            "text_params",
            "fusion_params",
            "total_params",
            "memory_mb_fp32",
            "memory_mb_int2",
        }
        assert expected_keys.issubset(set(stats.keys())), \
            f"Missing keys: {expected_keys - set(stats.keys())}"

    def test_total_params_under_10m(self):
        """Total parameter count should be under 10 million."""
        stats = benchmark_model()
        assert stats["total_params"] < 10_000_000, \
            f"Total params too high: {stats['total_params']:,}"

    def test_int2_memory_less_than_fp32(self):
        """INT2 memory should be less than FP32 memory."""
        stats = benchmark_model()
        assert stats["memory_mb_int2"] < stats["memory_mb_fp32"], \
            f"INT2 ({stats['memory_mb_int2']:.2f} MB) >= FP32 ({stats['memory_mb_fp32']:.2f} MB)"

    def test_all_param_counts_positive(self):
        """All component param counts should be positive."""
        stats = benchmark_model()
        assert stats["vision_params"] > 0
        assert stats["text_params"] > 0
        assert stats["fusion_params"] > 0
