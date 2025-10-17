import torch
from torch import nn
import torch.nn.functional as F
import math
from einops import rearrange


class T5RelativePositionBias(nn.Module):
    """
    Implements the T5 relative position bias mechanism.
    Instead of adding positional encodings to the input embeddings, T5 computes a bias term
    based on the relative distance between query and key pairs, which is then added to the attention scores.
    """

    def __init__(self, scale, causal=False, num_buckets=32, max_distance=128, heads=8):
        super().__init__()
        self.scale = scale
        self.causal = causal
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        # A learnable embedding table for the relative position buckets. Each head gets its own set of biases.
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, causal=True, num_buckets=32, max_distance=128):
        """
        Calculates the bucket index for a given relative position. This is a key part of T5's position handling.
        Distances are bucketed together, with closer distances having their own bucket and farther distances
        being grouped together logarithmically.
        """
        ret = 0
        n = -relative_position
        if not causal:
            # For bidirectional attention, we use half the buckets for positive and half for negative distances.
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            # For causal attention, we only consider past positions, so we clip negative relative positions to 0.
            n = torch.max(n, torch.zeros_like(n))

        # The first half of the buckets are for exact, small distances.
        max_exact = num_buckets // 2
        is_small = n < max_exact

        # Distances larger than max_exact are logarithmically bucketed.
        val_if_large = max_exact + (
                torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        # Combine the exact and bucketed values.
        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, qk_dots):
        # qk_dots: (batch, heads, seq_len_q, seq_len_k)
        seq_len_q, seq_len_k = qk_dots.shape[-2], qk_dots.shape[-1]
        device = qk_dots.device

        # Create position indices for query and key sequences.
        q_pos = torch.arange(seq_len_q, dtype=torch.long, device=device)
        k_pos = torch.arange(seq_len_k, dtype=torch.long, device=device)

        # Calculate the matrix of relative positions.
        rel_pos = k_pos[None, :] - q_pos[:, None]  # Shape: (seq_len_q, seq_len_k)

        # Compute the bucket for each relative position.
        rp_bucket = self._relative_position_bucket(
            rel_pos, causal=self.causal, num_buckets=self.num_buckets, max_distance=self.max_distance
        )

        # Lookup the bias values from the embedding table.
        values = self.relative_attention_bias(rp_bucket)  # Shape: (seq_len_q, seq_len_k, heads)

        # Rearrange to match the attention scores' shape for broadcasting.
        bias = rearrange(values, 'i j h -> h i j')  # Shape: (heads, seq_len_q, seq_len_k)

        # Add the bias to the attention scores.
        return qk_dots + bias


class T5LayerNorm(nn.Module):
    """
    Implements a T5-style Layer Normalization.
    It only has a learnable gamma (scale) parameter and no bias (beta), as is typical in T5 models.
    """

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))  # gamma is a learnable parameter

    def forward(self, x):
        # (batch, seq_len, dim) -> (batch, seq_len, dim)
        # The normalization is done over the last dimension (the feature dimension).
        return F.layer_norm(x, x.shape[-1:], self.gamma)


class FeedForward(nn.Module):
    """
    A standard Feed-Forward Network block for the transformer.
    It consists of two linear layers with a ReLU activation and dropout in between.
    """

    def __init__(self, dim, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, dim)
        )

    def forward(self, x):
        # (batch, seq_len, dim) -> (batch, seq_len, d_ff) -> (batch, seq_len, dim)
        return self.net(x)


class T5Attention(nn.Module):
    """
    Implements T5-style Multi-Head Attention, which can optionally use relative position bias.
    It handles self-attention, cross-attention, and causal masking.
    """

    def __init__(self, d_model, heads, causal=False, use_relative_bias=True, dropout=0.1, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5  # Scaling factor for the dot product
        self.causal = causal
        self.use_relative_bias = use_relative_bias

        self.to_q = nn.Linear(d_model, inner_dim, bias=False)  # Wq
        self.to_k = nn.Linear(d_model, inner_dim, bias=False)  # Wk
        self.to_v = nn.Linear(d_model, inner_dim, bias=False)  # Wv
        self.to_out = nn.Linear(inner_dim, d_model)  # Wo

        if self.use_relative_bias:
            self.relative_position_bias = T5RelativePositionBias(
                scale=self.scale, causal=causal, heads=heads
            )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context=None, mask=None, context_mask=None):
        if context is None:
            context = x  # Self-attention case

        # Project inputs to query, key, and value
        q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)

        # Reshape for multi-head attention: (batch, seq_len, inner_dim) -> (batch, heads, seq_len, dim_head)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))

        # Calculate attention scores (dot product)
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale  # (batch, heads, seq_len_q, seq_len_k)

        if self.use_relative_bias:
            sim = self.relative_position_bias(sim)

        mask_value = -torch.finfo(sim.dtype).max  # A very large negative number for masking

        if self.causal:
            # Apply causal mask to prevent attention to future tokens
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype=torch.bool, device=x.device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, mask_value)

        if mask is not None:
            # Apply padding mask for the query sequence
            sim = sim.masked_fill(mask.unsqueeze(1).unsqueeze(-1), mask_value)

        if context_mask is not None:
            # Apply padding mask for the key/value sequence (context)
            sim = sim.masked_fill(context_mask.unsqueeze(1).unsqueeze(2), mask_value)

        # Apply softmax to get attention weights
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        # Apply attention weights to the value vectors
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)

        # Reshape back to original format: (batch, heads, seq_len, dim_head) -> (batch, seq_len, inner_dim)
        out = rearrange(out, 'b h n d -> b n (h d)')

        # Final output projection
        return self.to_out(out)


class EncoderBlock(nn.Module):
    """
    A single block of the Transformer Encoder.
    It uses a pre-norm architecture: LayerNorm -> Attention -> Residual + Dropout, then LayerNorm -> FFN -> Residual + Dropout.
    """

    def __init__(self, d_model, heads, d_ff, dropout):
        super().__init__()
        self.attn = T5Attention(d_model, heads, causal=False, use_relative_bias=True, dropout=dropout,
                                dim_head=d_model // heads)
        self.ffn = FeedForward(d_model, d_ff, dropout=dropout)
        self.norm1 = T5LayerNorm(d_model)
        self.norm2 = T5LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # Self-attention part with residual connection
        x = x + self.dropout(self.attn(self.norm1(x), mask=mask))
        # Feed-forward part with residual connection
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class DecoderBlock(nn.Module):
    """
    A single block of the Transformer Decoder.
    It has three main sub-layers: self-attention, cross-attention, and a feed-forward network.
    It also uses a pre-norm architecture.
    """

    def __init__(self, d_model, heads, d_ff, dropout):
        super().__init__()
        dim_head = d_model // heads
        # Causal self-attention for the target sequence
        self.self_attn = T5Attention(d_model, heads, causal=True, use_relative_bias=True, dropout=dropout,
                                     dim_head=dim_head)
        # Cross-attention with the encoder's output
        self.cross_attn = T5Attention(d_model, heads, causal=False, use_relative_bias=False, dropout=dropout,
                                      dim_head=dim_head)
        self.ffn = FeedForward(d_model, d_ff, dropout=dropout)
        self.norm1 = T5LayerNorm(d_model)  # For self-attention
        self.norm2 = T5LayerNorm(d_model)  # For cross-attention
        self.norm3 = T5LayerNorm(d_model)  # For feed-forward network
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context, src_mask, tgt_mask):
        # Causal self-attention part
        x = x + self.dropout(self.self_attn(self.norm1(x), mask=tgt_mask))
        # Cross-attention part (attending to the encoder's output)
        x = x + self.dropout(self.cross_attn(self.norm2(x), context=context, context_mask=src_mask))
        # Feed-forward part
        x = x + self.dropout(self.ffn(self.norm3(x)))
        return x


class InputEmbeddings(nn.Module):
    """
    Converts input token IDs into dense vectors (embeddings).
    """

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.scale = math.sqrt(d_model)  # Scale the embeddings according to the paper

    def forward(self, x):
        # (batch, seq_len) -> (batch, seq_len, d_model)
        return self.embedding(x) * self.scale


class Transformer(nn.Module):
    """
    The main Transformer model, composed of an Encoder and a Decoder.
    It includes an option for 'multi_head' output, which creates two separate projection layers.
    """

    def __init__(self, vocab_size: int, d_model: int, N: int, h: int, d_ff: int, dropout: float,
                 multi_head: bool = False):
        super().__init__()
        self.multi_head = multi_head

        self.embedding = InputEmbeddings(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

        # Stack of N encoder blocks
        self.encoder_blocks = nn.ModuleList([EncoderBlock(d_model, h, d_ff, dropout) for _ in range(N)])
        # Stack of N decoder blocks
        self.decoder_blocks = nn.ModuleList([DecoderBlock(d_model, h, d_ff, dropout) for _ in range(N)])

        self.encoder_norm = T5LayerNorm(d_model)  # Final normalization for the encoder
        self.decoder_norm = T5LayerNorm(d_model)  # Final normalization for the decoder

        if self.multi_head:
            # If multi_head, create two separate output projection layers
            self.structured_projection_layer = nn.Linear(d_model, vocab_size)
            self.natural_language_projection_layer = nn.Linear(d_model, vocab_size)
        else:
            # A single projection layer to map decoder output to vocabulary
            self.projection_layer = nn.Linear(d_model, vocab_size)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initializes weights of the model using Xavier uniform initialization.
        """
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight)

    def encode(self, src: torch.Tensor, src_padding_mask: torch.Tensor):
        """
        Processes the source sequence through the encoder.
        """
        x = self.embedding(src)
        x = self.dropout(x)
        for block in self.encoder_blocks:
            x = block(x, src_padding_mask)
        return self.encoder_norm(x)

    def decode(self, encoder_output: torch.Tensor, src_padding_mask: torch.Tensor, tgt: torch.Tensor,
               tgt_padding_mask: torch.Tensor):
        """
        Processes the target sequence and encoder output through the decoder.
        """
        x = self.embedding(tgt)
        x = self.dropout(x)
        for block in self.decoder_blocks:
            x = block(x, encoder_output, src_padding_mask, tgt_padding_mask)
        return self.decoder_norm(x)


def build_transformer(vocab_size: int, seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1,
                      d_ff: int = 2048, multi_head: bool = False) -> Transformer:
    """
    A factory function to build and initialize the Transformer model.
    """
    model = Transformer(
        vocab_size=vocab_size, d_model=d_model, N=N, h=h, d_ff=d_ff, dropout=dropout, multi_head=multi_head
    )

    # Print model details for verification
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_type = "Multi-Head" if multi_head else "Single-Head"
    print(
        f"Modello Transformer ({model_type}, con T5 Relative Bias V2) costruito con {param_count:,} parametri.")
    return model
