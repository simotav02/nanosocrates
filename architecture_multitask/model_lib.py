import torch
from torch import nn
import torch.nn.functional as F
import math
from einops import rearrange


class T5RelativePositionBias(nn.Module):
    def __init__(self, scale, causal=False, num_buckets=32, max_distance=128, heads=8):
        super().__init__()
        self.scale = scale
        self.causal = causal
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, causal=True, num_buckets=32, max_distance=128):
        ret = 0
        n = -relative_position
        if not causal:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact
        val_if_large = max_exact + (
                torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, qk_dots):
        seq_len_q, seq_len_k = qk_dots.shape[-2], qk_dots.shape[-1]
        device = qk_dots.device
        q_pos = torch.arange(seq_len_q, dtype=torch.long, device=device)
        k_pos = torch.arange(seq_len_k, dtype=torch.long, device=device)
        rel_pos = k_pos[None, :] - q_pos[:, None]
        rp_bucket = self._relative_position_bucket(
            rel_pos, causal=self.causal, num_buckets=self.num_buckets, max_distance=self.max_distance
        )
        values = self.relative_attention_bias(rp_bucket)
        bias = rearrange(values, 'i j h -> h i j')
        return qk_dots + bias


class T5LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma)


class FeedForward(nn.Module):
    def __init__(self, dim, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, dim)
        )

    def forward(self, x):
        return self.net(x)


class T5Attention(nn.Module):
    def __init__(self, d_model, heads, causal=False, use_relative_bias=True, dropout=0.1, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.causal = causal
        self.use_relative_bias = use_relative_bias

        self.to_q = nn.Linear(d_model, inner_dim, bias=False)
        self.to_k = nn.Linear(d_model, inner_dim, bias=False)
        self.to_v = nn.Linear(d_model, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, d_model)

        if self.use_relative_bias:
            self.relative_position_bias = T5RelativePositionBias(
                scale=self.scale, causal=causal, heads=heads
            )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context=None, mask=None, context_mask=None):
        if context is None:
            context = x  # Self-attention

        q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))

        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        if self.use_relative_bias:
            sim = self.relative_position_bias(sim)

        mask_value = -torch.finfo(sim.dtype).max

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype=torch.bool, device=x.device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, mask_value)

        if mask is not None:
            sim = sim.masked_fill(mask.unsqueeze(1).unsqueeze(-1), mask_value)

        if context_mask is not None:
            sim = sim.masked_fill(context_mask.unsqueeze(1).unsqueeze(2), mask_value)

        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)


class EncoderBlock(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super().__init__()
        self.attn = T5Attention(d_model, heads, causal=False, use_relative_bias=True, dropout=dropout,
                                dim_head=d_model // heads)
        self.ffn = FeedForward(d_model, d_ff, dropout=dropout)
        self.norm1 = T5LayerNorm(d_model)
        self.norm2 = T5LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = x + self.dropout(self.attn(self.norm1(x), mask=mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super().__init__()
        dim_head = d_model // heads
        self.self_attn = T5Attention(d_model, heads, causal=True, use_relative_bias=True, dropout=dropout,
                                     dim_head=dim_head)
        self.cross_attn = T5Attention(d_model, heads, causal=False, use_relative_bias=False, dropout=dropout,
                                      dim_head=dim_head)
        self.ffn = FeedForward(d_model, d_ff, dropout=dropout)
        self.norm1 = T5LayerNorm(d_model)
        self.norm2 = T5LayerNorm(d_model)
        self.norm3 = T5LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context, src_mask, tgt_mask):
        x = x + self.dropout(self.self_attn(self.norm1(x), mask=tgt_mask))
        x = x + self.dropout(self.cross_attn(self.norm2(x), context=context, context_mask=src_mask))
        x = x + self.dropout(self.ffn(self.norm3(x)))
        return x


class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.scale = math.sqrt(d_model)

    def forward(self, x):
        return self.embedding(x) * self.scale


class Transformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, N: int, h: int, d_ff: int, dropout: float,
                 multi_head: bool = False):
        super().__init__()
        self.multi_head = multi_head

        self.embedding = InputEmbeddings(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.encoder_blocks = nn.ModuleList([EncoderBlock(d_model, h, d_ff, dropout) for _ in range(N)])
        self.decoder_blocks = nn.ModuleList([DecoderBlock(d_model, h, d_ff, dropout) for _ in range(N)])
        self.encoder_norm = T5LayerNorm(d_model)
        self.decoder_norm = T5LayerNorm(d_model)

        if self.multi_head:
            self.structured_projection_layer = nn.Linear(d_model, vocab_size)
            self.natural_language_projection_layer = nn.Linear(d_model, vocab_size)
        else:
            self.projection_layer = nn.Linear(d_model, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight)

    def encode(self, src: torch.Tensor, src_padding_mask: torch.Tensor):
        x = self.embedding(src)
        x = self.dropout(x)
        for block in self.encoder_blocks:
            x = block(x, src_padding_mask)
        return self.encoder_norm(x)

    def decode(self, encoder_output: torch.Tensor, src_padding_mask: torch.Tensor, tgt: torch.Tensor,
               tgt_padding_mask: torch.Tensor):
        x = self.embedding(tgt)
        x = self.dropout(x)
        for block in self.decoder_blocks:
            x = block(x, encoder_output, src_padding_mask, tgt_padding_mask)
        return self.decoder_norm(x)


def build_transformer(vocab_size: int, seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1,
                      d_ff: int = 2048, multi_head: bool = False) -> Transformer:
    model = Transformer(
        vocab_size=vocab_size, d_model=d_model, N=N, h=h, d_ff=d_ff, dropout=dropout, multi_head=multi_head
    )

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_type = "Multi-Head" if multi_head else "Single-Head"
    print(
        f"Modello Transformer ({model_type}, con T5 Relative Bias V2) costruito con {param_count:,} parametri.")
    return model