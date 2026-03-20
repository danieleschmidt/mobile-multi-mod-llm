"""TinyTextDecoder: 3-layer causal transformer decoder."""

import numpy as np

VOCAB_SIZE = 8192


class TinyTextDecoder:
    """
    Tiny text decoder with shared vocab embedding and causal attention.
    3-layer transformer decoder.
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        embed_dim: int = 256,
        n_heads: int = 4,
        n_layers: int = 3,
        max_len: int = 128,
    ):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_len = max_len

        rng = np.random.default_rng(123)
        scale = 0.02

        self.params = {}

        # Token embedding (shared with output projection)
        self.params["token_embed"] = rng.normal(0, scale, (vocab_size, embed_dim)).astype(np.float32)

        # Positional embedding
        self.params["pos_embed"] = rng.normal(0, scale, (max_len, embed_dim)).astype(np.float32)

        # Transformer layers
        for i in range(n_layers):
            # Self-attention
            self.params[f"layer{i}_W_q"] = rng.normal(0, scale, (embed_dim, embed_dim)).astype(np.float32)
            self.params[f"layer{i}_W_k"] = rng.normal(0, scale, (embed_dim, embed_dim)).astype(np.float32)
            self.params[f"layer{i}_W_v"] = rng.normal(0, scale, (embed_dim, embed_dim)).astype(np.float32)
            self.params[f"layer{i}_W_o"] = rng.normal(0, scale, (embed_dim, embed_dim)).astype(np.float32)
            self.params[f"layer{i}_ln_gamma"] = np.ones(embed_dim, dtype=np.float32)
            self.params[f"layer{i}_ln_beta"] = np.zeros(embed_dim, dtype=np.float32)
            # FFN
            self.params[f"layer{i}_ffn_W1"] = rng.normal(0, scale, (embed_dim, embed_dim * 4)).astype(np.float32)
            self.params[f"layer{i}_ffn_b1"] = np.zeros(embed_dim * 4, dtype=np.float32)
            self.params[f"layer{i}_ffn_W2"] = rng.normal(0, scale, (embed_dim * 4, embed_dim)).astype(np.float32)
            self.params[f"layer{i}_ffn_b2"] = np.zeros(embed_dim, dtype=np.float32)
            self.params[f"layer{i}_ln2_gamma"] = np.ones(embed_dim, dtype=np.float32)
            self.params[f"layer{i}_ln2_beta"] = np.zeros(embed_dim, dtype=np.float32)

        # Output layer norm
        self.params["out_ln_gamma"] = np.ones(embed_dim, dtype=np.float32)
        self.params["out_ln_beta"] = np.zeros(embed_dim, dtype=np.float32)

    def _layer_norm(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return gamma * (x - mean) / np.sqrt(var + eps) + beta

    def embed(self, token_ids) -> np.ndarray:
        """
        Embed token IDs with positional encoding.

        Args:
            token_ids: list or array of token ids, length seq_len

        Returns:
            (seq_len, embed_dim)
        """
        ids = np.asarray(token_ids, dtype=np.int32)
        seq_len = len(ids)
        tok_emb = self.params["token_embed"][ids]  # (seq_len, embed_dim)
        pos_emb = self.params["pos_embed"][:seq_len]
        return tok_emb + pos_emb

    def _causal_mask(self, seq_len: int) -> np.ndarray:
        """Create causal (lower-triangular) attention mask."""
        mask = np.triu(np.ones((seq_len, seq_len), dtype=np.float32), k=1)
        mask = mask * -1e9
        return mask  # (seq_len, seq_len)

    def attention(self, x: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """
        Multi-head causal self-attention (uses layer 0 weights; use _attention_layer for per-layer).

        Args:
            x: (seq_len, embed_dim)
            mask: optional (seq_len, seq_len) additive mask

        Returns:
            (seq_len, embed_dim)
        """
        return self._attention_layer(x, layer_idx=0, mask=mask)

    def _attention_layer(self, x: np.ndarray, layer_idx: int, mask: np.ndarray = None) -> np.ndarray:
        n, d = x.shape
        h = self.n_heads
        d_head = d // h

        W_q = self.params[f"layer{layer_idx}_W_q"]
        W_k = self.params[f"layer{layer_idx}_W_k"]
        W_v = self.params[f"layer{layer_idx}_W_v"]
        W_o = self.params[f"layer{layer_idx}_W_o"]

        Q = (x @ W_q).reshape(n, h, d_head).transpose(1, 0, 2)
        K = (x @ W_k).reshape(n, h, d_head).transpose(1, 0, 2)
        V = (x @ W_v).reshape(n, h, d_head).transpose(1, 0, 2)

        scale = np.sqrt(d_head)
        scores = Q @ K.transpose(0, 2, 1) / scale  # (h, n, n)

        if mask is not None:
            scores = scores + mask[np.newaxis, :, :]

        attn = np.exp(scores - scores.max(axis=-1, keepdims=True))
        attn = attn / (attn.sum(axis=-1, keepdims=True) + 1e-9)

        out = attn @ V  # (h, n, d_head)
        out = out.transpose(1, 0, 2).reshape(n, d)
        return out @ W_o

    def _ffn(self, x: np.ndarray, layer_idx: int) -> np.ndarray:
        W1 = self.params[f"layer{layer_idx}_ffn_W1"]
        b1 = self.params[f"layer{layer_idx}_ffn_b1"]
        W2 = self.params[f"layer{layer_idx}_ffn_W2"]
        b2 = self.params[f"layer{layer_idx}_ffn_b2"]
        h = x @ W1 + b1
        h = np.maximum(0, h)
        return h @ W2 + b2

    def forward(self, token_ids) -> np.ndarray:
        """
        Forward pass through all decoder layers.

        Args:
            token_ids: list or array of token ids

        Returns:
            (seq_len, vocab_size) logits
        """
        x = self.embed(token_ids)  # (seq_len, embed_dim)
        seq_len = x.shape[0]
        mask = self._causal_mask(seq_len)

        for i in range(self.n_layers):
            # Self-attention with residual
            residual = x
            x = self._layer_norm(x, self.params[f"layer{i}_ln_gamma"], self.params[f"layer{i}_ln_beta"])
            x = self._attention_layer(x, layer_idx=i, mask=mask)
            x = x + residual

            # FFN with residual
            residual = x
            x = self._layer_norm(x, self.params[f"layer{i}_ln2_gamma"], self.params[f"layer{i}_ln2_beta"])
            x = self._ffn(x, layer_idx=i)
            x = x + residual

        # Output layer norm + project to vocab (shared embedding)
        x = self._layer_norm(x, self.params["out_ln_gamma"], self.params["out_ln_beta"])
        logits = x @ self.params["token_embed"].T  # (seq_len, vocab_size)
        return logits

    def param_count(self) -> int:
        """Total number of parameters (token_embed counted once)."""
        return sum(arr.size for arr in self.params.values())

    def greedy_decode(self, context_embed: np.ndarray, max_new_tokens: int = 20) -> list:
        """
        Greedy decoding using context embedding as a prefix.

        Args:
            context_embed: (ctx_len, embed_dim) vision/context features
            max_new_tokens: max tokens to generate

        Returns:
            list of generated token ids
        """
        # Start with BOS token (id=1)
        generated = [1]
        ctx_len = context_embed.shape[0]

        for _ in range(max_new_tokens):
            # Get embeddings for generated tokens so far
            tok_emb = self.embed(generated)  # (gen_len, embed_dim)

            # Concatenate context with generated embeddings
            x = np.concatenate([context_embed, tok_emb], axis=0)  # (ctx_len + gen_len, embed_dim)
            total_len = x.shape[0]

            # Apply causal mask
            mask = self._causal_mask(total_len)

            for i in range(self.n_layers):
                residual = x
                x = self._layer_norm(x, self.params[f"layer{i}_ln_gamma"], self.params[f"layer{i}_ln_beta"])
                x = self._attention_layer(x, layer_idx=i, mask=mask)
                x = x + residual

                residual = x
                x = self._layer_norm(x, self.params[f"layer{i}_ln2_gamma"], self.params[f"layer{i}_ln2_beta"])
                x = self._ffn(x, layer_idx=i)
                x = x + residual

            x = self._layer_norm(x, self.params["out_ln_gamma"], self.params["out_ln_beta"])
            logits = x @ self.params["token_embed"].T  # (total_len, vocab_size)

            # Predict next token from last position
            next_logits = logits[-1]  # (vocab_size,)
            next_token = int(np.argmax(next_logits))
            generated.append(next_token)

            # Stop at EOS (id=2)
            if next_token == 2:
                break

        return generated[1:]  # skip BOS
