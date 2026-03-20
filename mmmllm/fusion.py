"""CrossModalFusion: cross-attention between vision and text features."""

import numpy as np


class CrossModalFusion:
    """
    Cross-modal fusion via cross-attention.
    Text queries attend to vision key/values.
    """

    def __init__(self, embed_dim: int = 256, n_heads: int = 4):
        self.embed_dim = embed_dim
        self.n_heads = n_heads

        rng = np.random.default_rng(77)
        scale = 0.02

        self.params = {}

        # Cross-attention projections
        self.params["W_q"] = rng.normal(0, scale, (embed_dim, embed_dim)).astype(np.float32)
        self.params["W_k"] = rng.normal(0, scale, (embed_dim, embed_dim)).astype(np.float32)
        self.params["W_v"] = rng.normal(0, scale, (embed_dim, embed_dim)).astype(np.float32)
        self.params["W_o"] = rng.normal(0, scale, (embed_dim, embed_dim)).astype(np.float32)

        # Layer norm for query and output
        self.params["ln_q_gamma"] = np.ones(embed_dim, dtype=np.float32)
        self.params["ln_q_beta"] = np.zeros(embed_dim, dtype=np.float32)
        self.params["ln_out_gamma"] = np.ones(embed_dim, dtype=np.float32)
        self.params["ln_out_beta"] = np.zeros(embed_dim, dtype=np.float32)

        # FFN after cross-attention
        self.params["ffn_W1"] = rng.normal(0, scale, (embed_dim, embed_dim * 2)).astype(np.float32)
        self.params["ffn_b1"] = np.zeros(embed_dim * 2, dtype=np.float32)
        self.params["ffn_W2"] = rng.normal(0, scale, (embed_dim * 2, embed_dim)).astype(np.float32)
        self.params["ffn_b2"] = np.zeros(embed_dim, dtype=np.float32)

    def _layer_norm(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return gamma * (x - mean) / np.sqrt(var + eps) + beta

    def cross_attention(self, query: np.ndarray, key_value: np.ndarray) -> np.ndarray:
        """
        Cross-attention: query attends to key_value.

        Args:
            query: (q_len, embed_dim) - text features
            key_value: (kv_len, embed_dim) - vision features

        Returns:
            (q_len, embed_dim)
        """
        q_len, d = query.shape
        kv_len = key_value.shape[0]
        h = self.n_heads
        d_head = d // h

        Q = (query @ self.params["W_q"]).reshape(q_len, h, d_head).transpose(1, 0, 2)    # (h, q_len, d_head)
        K = (key_value @ self.params["W_k"]).reshape(kv_len, h, d_head).transpose(1, 0, 2)  # (h, kv_len, d_head)
        V = (key_value @ self.params["W_v"]).reshape(kv_len, h, d_head).transpose(1, 0, 2)  # (h, kv_len, d_head)

        scale = np.sqrt(d_head)
        scores = Q @ K.transpose(0, 2, 1) / scale  # (h, q_len, kv_len)

        attn = np.exp(scores - scores.max(axis=-1, keepdims=True))
        attn = attn / (attn.sum(axis=-1, keepdims=True) + 1e-9)

        out = attn @ V  # (h, q_len, d_head)
        out = out.transpose(1, 0, 2).reshape(q_len, d)  # (q_len, d)
        return out @ self.params["W_o"]

    def forward(self, text_features: np.ndarray, vision_features: np.ndarray) -> np.ndarray:
        """
        Fuse text and vision features via cross-attention.

        Args:
            text_features: (seq_len, embed_dim)
            vision_features: (patches, embed_dim)

        Returns:
            (seq_len, embed_dim) fused features
        """
        # Normalize query
        q_norm = self._layer_norm(text_features, self.params["ln_q_gamma"], self.params["ln_q_beta"])

        # Cross-attention with residual
        attn_out = self.cross_attention(q_norm, vision_features)
        x = text_features + attn_out

        # FFN with residual
        residual = x
        x = self._layer_norm(x, self.params["ln_out_gamma"], self.params["ln_out_beta"])
        h = x @ self.params["ffn_W1"] + self.params["ffn_b1"]
        h = np.maximum(0, h)  # ReLU
        x = h @ self.params["ffn_W2"] + self.params["ffn_b2"]
        x = x + residual

        return x  # (seq_len, embed_dim)

    def param_count(self) -> int:
        """Total number of parameters."""
        return sum(arr.size for arr in self.params.values())
