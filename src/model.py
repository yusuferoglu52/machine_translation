"""
Transformer architecture for Neural Machine Translation (English → German).

Implements the original "Attention Is All You Need" (Vaswani et al., 2017)
architecture from scratch — no nn.Transformer usage.

Components:
    MultiHeadAttention      — Scaled dot-product attention across h heads
    PositionalEncoding      — Sinusoidal position embeddings (fixed, not learned)
    PositionWiseFeedForward — Two-layer FFN with GELU activation
    EncoderLayer            — MHA + FFN with pre-LN residual connections
    DecoderLayer            — Masked MHA + cross-MHA + FFN with pre-LN
    Encoder / Decoder       — Stacks of N layers
    Seq2SeqTransformer      — Full encoder-decoder model with weight tying
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Scaled Dot-Product Attention (single head, batched)
# ---------------------------------------------------------------------------

def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
    dropout: nn.Dropout | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    Args:
        query: (batch, heads, seq_q, d_k)
        key:   (batch, heads, seq_k, d_k)
        value: (batch, heads, seq_k, d_v)
        mask:  broadcastable bool tensor; True positions are masked out
        dropout: optional nn.Dropout applied to attention weights

    Returns:
        context: (batch, heads, seq_q, d_v)
        attn_weights: (batch, heads, seq_q, seq_k)
    """
    d_k = query.size(-1)
    # (batch, heads, seq_q, seq_k)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))

    # Replace -inf mask positions with a large-but-finite negative value before
    # softmax.  Using -inf causes 0/0 = NaN in the softmax for fully-masked rows
    # (e.g. PAD-only decoder positions), and nan_to_num in the forward pass fixes
    # the NaN value but does not detach the NaN from the autograd graph — so the
    # backward still propagates NaN gradients, corrupting the model on step 1.
    # Clamping to -1e4 (well above FP16 underflow, << any real score) keeps the
    # softmax output a valid probability distribution without NaN anywhere.
    scores = scores.clamp(min=-1e4)

    attn_weights = F.softmax(scores, dim=-1)

    if dropout is not None:
        attn_weights = dropout(attn_weights)

    context = torch.matmul(attn_weights, value)
    return context, attn_weights


# ---------------------------------------------------------------------------
# Multi-Head Attention
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention as in Vaswani et al., 2017.

    Projects Q, K, V into h subspaces of dimension d_k = d_model // h,
    runs scaled dot-product attention in parallel, then concatenates and
    projects back to d_model.

        MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
        head_i = Attention(Q W^Q_i, K W^K_i, V W^V_i)

    Args:
        d_model:  Model dimensionality.
        n_heads:  Number of attention heads.
        dropout:  Dropout probability on attention weights.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Fused projections: single matrix for all heads, split after
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.attn_weights: torch.Tensor | None = None  # stored for inspection

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, seq, d_model) → (batch, heads, seq, d_k)"""
        batch, seq, _ = x.size()
        return x.view(batch, seq, self.n_heads, self.d_k).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, heads, seq, d_k) → (batch, seq, d_model)"""
        batch, _, seq, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch, seq, self.d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            query: (batch, seq_q, d_model)
            key:   (batch, seq_k, d_model)
            value: (batch, seq_k, d_model)
            mask:  (batch, 1, seq_q, seq_k) or (batch, 1, 1, seq_k) — bool,
                   True = masked out.

        Returns:
            output: (batch, seq_q, d_model)
        """
        q = self._split_heads(self.w_q(query))  # (B, H, S_q, d_k)
        k = self._split_heads(self.w_k(key))    # (B, H, S_k, d_k)
        v = self._split_heads(self.w_v(value))  # (B, H, S_k, d_k)

        context, self.attn_weights = scaled_dot_product_attention(
            q, k, v, mask=mask, dropout=self.attn_dropout
        )

        output = self.w_o(self._merge_heads(context))
        return output


# ---------------------------------------------------------------------------
# Position-Wise Feed-Forward Network
# ---------------------------------------------------------------------------

class PositionWiseFeedForward(nn.Module):
    """
    FFN(x) = max(0, xW_1 + b_1)W_2 + b_2

    Paper uses ReLU; we use GELU which empirically outperforms ReLU in
    language tasks (Hendrycks & Gimpel, 2016).

    Args:
        d_model:  Input/output dimensionality.
        d_ff:     Inner layer dimensionality (typically 4 × d_model).
        dropout:  Dropout after the activation.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


# ---------------------------------------------------------------------------
# Sinusoidal Positional Encoding
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """
    Injects position information via fixed sinusoidal encodings:

        PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

    The encoding is precomputed up to max_len and added to the embedding.

    Args:
        d_model:  Embedding dimensionality.
        dropout:  Dropout applied after adding the positional encoding.
        max_len:  Maximum sequence length to precompute.
    """

    def __init__(
        self, d_model: int, dropout: float = 0.1, max_len: int = 5000
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Build encoding matrix: (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer: moves with .to(device) but not a parameter
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq, d_model) — embedding tensor

        Returns:
            (batch, seq, d_model) with positional encoding added
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Encoder Layer (Pre-LN variant for training stability)
# ---------------------------------------------------------------------------

class EncoderLayer(nn.Module):
    """
    Single Transformer encoder block using Pre-Layer Normalization:

        x = x + MHA(LN(x), LN(x), LN(x))
        x = x + FFN(LN(x))

    Pre-LN (vs Post-LN from the original paper) is used because it
    stabilizes training without requiring careful learning rate warm-up.

    Args:
        d_model:  Model dimensionality.
        n_heads:  Number of attention heads.
        d_ff:     Feed-forward inner dimensionality.
        dropout:  Dropout probability.
    """

    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, src_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x:        (batch, src_len, d_model)
            src_mask: (batch, 1, 1, src_len) — True for PAD positions

        Returns:
            (batch, src_len, d_model)
        """
        # Self-attention sublayer
        x = x + self.dropout(self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), src_mask))
        # FFN sublayer
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# Decoder Layer
# ---------------------------------------------------------------------------

class DecoderLayer(nn.Module):
    """
    Single Transformer decoder block with three sublayers:

        1. Masked self-attention (causal, prevents attending to future tokens)
        2. Cross-attention over encoder output
        3. Position-wise FFN

    All three use Pre-LN residual connections.

    Args:
        d_model:  Model dimensionality.
        n_heads:  Number of attention heads.
        d_ff:     Feed-forward inner dimensionality.
        dropout:  Dropout probability.
    """

    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        src_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:        (batch, tgt_len, d_model) — decoder input
            memory:   (batch, src_len, d_model) — encoder output
            tgt_mask: (batch, 1, tgt_len, tgt_len) — causal + PAD mask
            src_mask: (batch, 1, 1,       src_len) — encoder PAD mask

        Returns:
            (batch, tgt_len, d_model)
        """
        # 1. Masked self-attention
        x_norm = self.norm1(x)
        x = x + self.dropout(self.self_attn(x_norm, x_norm, x_norm, tgt_mask))

        # 2. Cross-attention (query from decoder, key/value from encoder)
        x_norm = self.norm2(x)
        x = x + self.dropout(self.cross_attn(x_norm, memory, memory, src_mask))

        # 3. FFN
        x = x + self.dropout(self.ffn(self.norm3(x)))
        return x


# ---------------------------------------------------------------------------
# Encoder Stack
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """Stack of N EncoderLayers with a final LayerNorm (Pre-LN convention)."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float,
        max_len: int = 5000,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len)
        self.scale = math.sqrt(d_model)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, src: torch.Tensor, src_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            src:      (batch, src_len) — token ids
            src_mask: (batch, 1, 1, src_len)

        Returns:
            memory: (batch, src_len, d_model)
        """
        x = self.pos_encoding(self.embedding(src) * self.scale)
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)


# ---------------------------------------------------------------------------
# Decoder Stack
# ---------------------------------------------------------------------------

class Decoder(nn.Module):
    """Stack of N DecoderLayers with a final LayerNorm (Pre-LN convention)."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float,
        max_len: int = 5000,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len)
        self.scale = math.sqrt(d_model)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        src_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            tgt:      (batch, tgt_len) — token ids
            memory:   (batch, src_len, d_model) — encoder output
            tgt_mask: (batch, 1, tgt_len, tgt_len)
            src_mask: (batch, 1, 1, src_len)

        Returns:
            (batch, tgt_len, d_model)
        """
        x = self.pos_encoding(self.embedding(tgt) * self.scale)
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, src_mask)
        return self.norm(x)


# ---------------------------------------------------------------------------
# Full Seq2Seq Transformer
# ---------------------------------------------------------------------------

class Seq2SeqTransformer(nn.Module):
    """
    Encoder-Decoder Transformer for sequence-to-sequence translation.

    Supports shared source/target vocabulary (weight tying between encoder
    embedding, decoder embedding, and output projection — reduces parameters
    and improves translation quality when src/tgt share a joint BPE vocab).

    Args:
        src_vocab_size:   Source vocabulary size.
        tgt_vocab_size:   Target vocabulary size.
        d_model:          Model dimensionality (default 512).
        n_heads:          Attention heads (default 8).
        n_encoder_layers: Encoder depth (default 6).
        n_decoder_layers: Decoder depth (default 6).
        d_ff:             FFN inner dim (default 2048).
        dropout:          Dropout probability (default 0.1).
        max_len:          Max sequence length for positional encoding.
        share_embeddings: Tie src embedding, tgt embedding, output weights.
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_encoder_layers: int = 6,
        n_decoder_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_len: int = 5000,
        share_embeddings: bool = True,
    ) -> None:
        super().__init__()

        self.encoder = Encoder(
            src_vocab_size, d_model, n_heads, n_encoder_layers, d_ff, dropout, max_len
        )
        self.decoder = Decoder(
            tgt_vocab_size, d_model, n_heads, n_decoder_layers, d_ff, dropout, max_len
        )
        # Output projection: (batch, tgt_len, d_model) → (batch, tgt_len, tgt_vocab)
        self.output_projection = nn.Linear(d_model, tgt_vocab_size, bias=False)

        if share_embeddings:
            # Weight tying: all three matrices share the same parameters
            # Requires a joint/shared BPE vocabulary
            assert src_vocab_size == tgt_vocab_size, (
                "Weight tying requires src_vocab_size == tgt_vocab_size. "
                "Train a shared BPE vocabulary over both languages."
            )
            self.decoder.embedding.weight = self.encoder.embedding.weight
            self.output_projection.weight = self.encoder.embedding.weight

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier uniform initialization for all linear/embedding parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(
        self, src: torch.Tensor, src_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.encoder(src, src_mask)

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        src_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.decoder(tgt, memory, tgt_mask, src_mask)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            src:      (batch, src_len) — source token ids
            tgt:      (batch, tgt_len) — target token ids (teacher-forced)
            src_mask: (batch, 1, 1, src_len)       — PAD mask for encoder
            tgt_mask: (batch, 1, tgt_len, tgt_len) — causal + PAD mask

        Returns:
            logits: (batch, tgt_len, tgt_vocab_size)
        """
        memory = self.encode(src, src_mask)
        decoder_out = self.decode(tgt, memory, tgt_mask, src_mask)
        return self.output_projection(decoder_out)


# ---------------------------------------------------------------------------
# Mask Utilities
# ---------------------------------------------------------------------------

def make_src_mask(src: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """
    Creates a padding mask for the encoder.

    Args:
        src:     (batch, src_len) — token ids
        pad_idx: ID of the [PAD] token

    Returns:
        (batch, 1, 1, src_len) bool tensor — True where PAD
    """
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """
    Creates a combined causal + padding mask for the decoder.

    The causal (look-ahead) mask prevents position i from attending to j > i.
    The padding mask prevents attention to [PAD] tokens.

    Args:
        tgt:     (batch, tgt_len) — token ids
        pad_idx: ID of the [PAD] token

    Returns:
        (batch, 1, tgt_len, tgt_len) bool tensor — True where masked
    """
    tgt_len = tgt.size(1)

    # Causal mask: upper triangle (excluding diagonal) is True
    causal_mask = torch.triu(
        torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=tgt.device),
        diagonal=1,
    )  # (tgt_len, tgt_len)

    # PAD mask: True where the target token is PAD
    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, tgt_len)

    # Broadcast both to (batch, 1, tgt_len, tgt_len)
    return causal_mask.unsqueeze(0).unsqueeze(0) | pad_mask


# ---------------------------------------------------------------------------
# Parameter Count Helper
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    """Returns the total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Quick Sanity Check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    VOCAB_SIZE = 10_000
    D_MODEL = 512
    N_HEADS = 8
    N_LAYERS = 6
    D_FF = 2048
    BATCH = 4
    SRC_LEN = 32
    TGT_LEN = 28
    PAD_IDX = 0

    model = Seq2SeqTransformer(
        src_vocab_size=VOCAB_SIZE,
        tgt_vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_encoder_layers=N_LAYERS,
        n_decoder_layers=N_LAYERS,
        d_ff=D_FF,
        share_embeddings=True,
    ).to(device)

    print(f"Parameters: {count_parameters(model):,}")

    src = torch.randint(1, VOCAB_SIZE, (BATCH, SRC_LEN), device=device)
    tgt = torch.randint(1, VOCAB_SIZE, (BATCH, TGT_LEN), device=device)
    src[:, -3:] = PAD_IDX  # inject some padding
    tgt[:, -2:] = PAD_IDX

    src_mask = make_src_mask(src, PAD_IDX)
    tgt_mask = make_tgt_mask(tgt, PAD_IDX)

    with torch.no_grad():
        logits = model(src, tgt, src_mask, tgt_mask)

    print(f"Input  src shape : {src.shape}")
    print(f"Input  tgt shape : {tgt.shape}")
    print(f"Output logits    : {logits.shape}")   # (BATCH, TGT_LEN, VOCAB_SIZE)
    assert logits.shape == (BATCH, TGT_LEN, VOCAB_SIZE), "Shape mismatch!"
    print("Sanity check passed.")
