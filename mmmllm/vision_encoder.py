"""TinyVisionEncoder: patch embedding + multi-head attention vision encoder."""

import numpy as np


class TinyVisionEncoder:
    """
    Tiny vision encoder with patch embedding and multi-layer attention.
    ~5M parameter budget.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
    ):
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_layers = n_layers

        self.n_patches = (img_size // patch_size) ** 2
        self.patch_dim = patch_size * patch_size * 3  # RGB patches

        rng = np.random.default_rng(42)
        scale = 0.02

        self.params = {}

        # Patch embedding projection: (patch_dim, embed_dim)
        self.params["patch_proj_W"] = rng.normal(0, scale, (self.patch_dim, embed_dim)).astype(np.float32)
        self.params["patch_proj_b"] = np.zeros(embed_dim, dtype=np.float32)

        # CLS token
        self.params["cls_token"] = rng.normal(0, scale, (1, embed_dim)).astype(np.float32)

        # Positional embeddings: (n_patches + 1, embed_dim)
        self.params["pos_embed"] = rng.normal(0, scale, (self.n_patches + 1, embed_dim)).astype(np.float32)

        # Per-layer attention weights
        for i in range(n_layers):
            # Q, K, V projections
            self.params[f"layer{i}_W_q"] = rng.normal(0, scale, (embed_dim, embed_dim)).astype(np.float32)
            self.params[f"layer{i}_W_k"] = rng.normal(0, scale, (embed_dim, embed_dim)).astype(np.float32)
            self.params[f"layer{i}_W_v"] = rng.normal(0, scale, (embed_dim, embed_dim)).astype(np.float32)
            self.params[f"layer{i}_W_o"] = rng.normal(0, scale, (embed_dim, embed_dim)).astype(np.float32)
            # LayerNorm params
            self.params[f"layer{i}_ln_gamma"] = np.ones(embed_dim, dtype=np.float32)
            self.params[f"layer{i}_ln_beta"] = np.zeros(embed_dim, dtype=np.float32)
            # FFN
            self.params[f"layer{i}_ffn_W1"] = rng.normal(0, scale, (embed_dim, embed_dim * 4)).astype(np.float32)
            self.params[f"layer{i}_ffn_b1"] = np.zeros(embed_dim * 4, dtype=np.float32)
            self.params[f"layer{i}_ffn_W2"] = rng.normal(0, scale, (embed_dim * 4, embed_dim)).astype(np.float32)
            self.params[f"layer{i}_ffn_b2"] = np.zeros(embed_dim, dtype=np.float32)
            self.params[f"layer{i}_ln2_gamma"] = np.ones(embed_dim, dtype=np.float32)
            self.params[f"layer{i}_ln2_beta"] = np.zeros(embed_dim, dtype=np.float32)

    def patch_embed(self, image: np.ndarray) -> np.ndarray:
        """
        Split image into patches and project to embed_dim.

        Args:
            image: (H, W, 3) or (H, W) numpy array

        Returns:
            (n_patches, embed_dim)
        """
        if image.ndim == 2:
            # Grayscale -> RGB
            image = np.stack([image, image, image], axis=-1)

        H, W, C = image.shape
        ps = self.patch_size

        # Pad image if needed
        H_pad = ((H + ps - 1) // ps) * ps
        W_pad = ((W + ps - 1) // ps) * ps
        if H_pad != H or W_pad != W:
            padded = np.zeros((H_pad, W_pad, C), dtype=image.dtype)
            padded[:H, :W, :] = image
            image = padded
            H, W = H_pad, W_pad

        n_h = H // ps
        n_w = W // ps

        # Extract patches: (n_h, n_w, ps, ps, C)
        patches = image.reshape(n_h, ps, n_w, ps, C)
        patches = patches.transpose(0, 2, 1, 3, 4)  # (n_h, n_w, ps, ps, C)
        patches = patches.reshape(-1, ps * ps * C).astype(np.float32)  # (n_patches, patch_dim)

        # Linear projection
        out = patches @ self.params["patch_proj_W"] + self.params["patch_proj_b"]
        return out  # (n_patches, embed_dim)

    def _layer_norm(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return gamma * (x - mean) / np.sqrt(var + eps) + beta

    def attention(
        self,
        x: np.ndarray,
        W_q: np.ndarray,
        W_k: np.ndarray,
        W_v: np.ndarray,
        W_o: np.ndarray,
        mask: np.ndarray = None,
    ) -> np.ndarray:
        """
        Multi-head self-attention.

        Args:
            x: (n, embed_dim)
            W_q, W_k, W_v, W_o: (embed_dim, embed_dim)

        Returns:
            (n, embed_dim)
        """
        n, d = x.shape
        h = self.n_heads
        d_head = d // h

        Q = x @ W_q  # (n, d)
        K = x @ W_k
        V = x @ W_v

        # Reshape to multi-head
        Q = Q.reshape(n, h, d_head).transpose(1, 0, 2)  # (h, n, d_head)
        K = K.reshape(n, h, d_head).transpose(1, 0, 2)
        V = V.reshape(n, h, d_head).transpose(1, 0, 2)

        # Scaled dot-product attention
        scale = np.sqrt(d_head)
        scores = Q @ K.transpose(0, 2, 1) / scale  # (h, n, n)

        if mask is not None:
            scores = scores + mask

        attn = np.exp(scores - scores.max(axis=-1, keepdims=True))
        attn = attn / (attn.sum(axis=-1, keepdims=True) + 1e-9)

        out = attn @ V  # (h, n, d_head)
        out = out.transpose(1, 0, 2).reshape(n, d)  # (n, d)
        return out @ W_o

    def _ffn(self, x: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
        h = x @ W1 + b1
        h = np.maximum(0, h)  # ReLU
        return h @ W2 + b2

    def forward(self, image: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        Args:
            image: (H, W, 3) or (H, W) numpy array

        Returns:
            (n_patches+1, embed_dim) with CLS token prepended
        """
        x = self.patch_embed(image)  # (n_patches, embed_dim)

        # Prepend CLS token
        cls = self.params["cls_token"]  # (1, embed_dim)
        x = np.concatenate([cls, x], axis=0)  # (n_patches+1, embed_dim)

        # Add positional embeddings
        n = x.shape[0]
        x = x + self.params["pos_embed"][:n]

        # Transformer layers
        for i in range(self.n_layers):
            W_q = self.params[f"layer{i}_W_q"]
            W_k = self.params[f"layer{i}_W_k"]
            W_v = self.params[f"layer{i}_W_v"]
            W_o = self.params[f"layer{i}_W_o"]

            # Self-attention with residual
            residual = x
            x = self._layer_norm(x, self.params[f"layer{i}_ln_gamma"], self.params[f"layer{i}_ln_beta"])
            x = self.attention(x, W_q, W_k, W_v, W_o)
            x = x + residual

            # FFN with residual
            residual = x
            x = self._layer_norm(x, self.params[f"layer{i}_ln2_gamma"], self.params[f"layer{i}_ln2_beta"])
            x = self._ffn(
                x,
                self.params[f"layer{i}_ffn_W1"],
                self.params[f"layer{i}_ffn_b1"],
                self.params[f"layer{i}_ffn_W2"],
                self.params[f"layer{i}_ffn_b2"],
            )
            x = x + residual

        return x  # (n_patches+1, embed_dim)

    def param_count(self) -> int:
        """Total number of parameters."""
        return sum(arr.size for arr in self.params.values())
