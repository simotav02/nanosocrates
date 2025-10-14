# ==============================================================================
# model_lib.py (Versione Finale per Confronto Diretto)
#
# Questo file offre due architetture per un confronto scientifico:
#   1. 'relative_bias': Il tuo modello T5-style originale (baseline).
#   2. 'mla_rope_decoupled': L'architettura avanzata del Prof. Anelli, che
#      fonde nativamente Multi-Latent Attention e RoPE in modo disaccoppiato.
#
# L'architettura viene scelta tramite il parametro `attention_type_str`
# nella funzione `build_transformer`.
# ==============================================================================

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from dataclasses import dataclass
from enum import Enum
import math
from einops import rearrange

# --- CONFIGURAZIONE E CLASSI HELPER ---

class AttentionType(Enum):
    RELATIVE_BIAS = "relative_bias"
    MLA_ROPE_DECOUPLED = "mla_rope_decoupled"

@dataclass
class ModelArgs:
    """Configurazione del modello per il layer di attenzione avanzato."""
    dim: int
    n_heads: int
    n_kv_heads: int = 4 # Default ragionevole per GQA, puoi renderlo configurabile
    rope_theta: float = 10000
    max_position_embeddings: int = 4096
    # Parametri specifici MLA-RoPE
    q_compressed_dim: int = 128
    q_nope_head_dim: int = 96
    q_rope_head_dim: int = 32
    kv_compressed_dim: int = 128
    k_nope_head_dim: int = 64
    k_rope_head_dim: int = 64
    v_head_dim: int = 256

    @property
    def gqa_factor(self) -> int: return self.n_heads // self.n_kv_heads

# --- MODULI FONDAMENTALI ---

def repeat_kv_heads(x: Tensor, n_rep: int) -> Tensor:
    if n_rep == 1: return x
    batch, n_kv_heads, seq_len, head_dim = x.shape
    return x.unsqueeze(2).expand(batch, n_kv_heads, n_rep, seq_len, head_dim).reshape(batch, n_kv_heads * n_rep, seq_len, head_dim)

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, args: ModelArgs, head_dim: int):
        super().__init__()
        if head_dim % 2: raise ValueError("head_dim must be even")
        inv_freq = 1.0 / (args.rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(args.max_position_embeddings, dtype=torch.float32)
        theta = torch.outer(positions, inv_freq)
        self.register_buffer("cos_cached", theta.cos(), persistent=False)
        self.register_buffer("sin_cached", theta.sin(), persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        cos = self.cos_cached[position_ids, :].unsqueeze(1).to(dtype=x.dtype)
        sin = self.sin_cached[position_ids, :].unsqueeze(1).to(dtype=x.dtype)
        x_even, x_odd = x[..., ::2], x[..., 1::2]
        rot_even = x_even * cos - x_odd * sin
        rot_odd = x_even * sin + x_odd * cos
        return torch.stack((rot_even, rot_odd), dim=-1).flatten(-2)

# --- I TUOI COMPONENTI ORIGINALI (MANTENUTI PER LA BASELINE) ---
class T5LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return nn.functional.layer_norm(x, x.shape[-1:], self.gamma)

class FeedForward(nn.Module):
    def __init__(self, dim, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, dim))
    def forward(self, x): return self.net(x)

# --- MODULI DI ATTENZIONE ---

# 1. Baseline: Attenzione T5 originale con Bias Relativo
class T5RelativePositionBias(nn.Module):
    def __init__(self, scale, causal=False, num_buckets=32, max_distance=128, heads=8):
        super().__init__()
        self.scale, self.causal, self.num_buckets, self.max_distance = scale, causal, num_buckets, max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, causal=True, num_buckets=32, max_distance=128):
        ret, n = 0, -relative_position
        if not causal:
            num_buckets //= 2; ret += (n < 0).long() * num_buckets; n = torch.abs(n)
        else: n = torch.max(n, torch.zeros_like(n))
        max_exact = num_buckets // 2
        is_small = n < max_exact
        val_if_large = max_exact + (torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, qk_dots):
        seq_len_q, seq_len_k = qk_dots.shape[-2], qk_dots.shape[-1]
        q_pos = torch.arange(seq_len_q, dtype=torch.long, device=qk_dots.device)
        k_pos = torch.arange(seq_len_k, dtype=torch.long, device=qk_dots.device)
        rel_pos = k_pos[None, :] - q_pos[:, None]
        rp_bucket = self._relative_position_bucket(rel_pos, self.causal, self.num_buckets, self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        return qk_dots + rearrange(values, 'i j h -> h i j')

class T5Attention(nn.Module):
    def __init__(self, d_model, heads, causal=False, use_relative_bias=True, dropout=0.1):
        super().__init__()
        dim_head = d_model // heads; inner_dim = dim_head * heads
        self.heads, self.scale, self.causal, self.use_relative_bias = heads, dim_head ** -0.5, causal, use_relative_bias
        self.to_q, self.to_k, self.to_v = (nn.Linear(d_model, inner_dim, bias=False) for _ in range(3))
        self.to_out = nn.Linear(inner_dim, d_model)
        if self.use_relative_bias: self.relative_position_bias = T5RelativePositionBias(self.scale, causal, heads=heads)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context=None, mask=None, context_mask=None, **kwargs):
        if context is None: context = x
        q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        if self.use_relative_bias: sim = self.relative_position_bias(sim)
        mask_value = -torch.finfo(sim.dtype).max
        if self.causal:
            i, j = sim.shape[-2:]; causal_mask = torch.ones((i, j), dtype=torch.bool, device=x.device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, mask_value)
        if mask is not None: sim = sim.masked_fill(mask.unsqueeze(1).unsqueeze(-1), mask_value)
        if context_mask is not None: sim = sim.masked_fill(context_mask.unsqueeze(1).unsqueeze(2), mask_value)
        attn = self.dropout(sim.softmax(dim=-1))
        out = rearrange(torch.einsum('b h i j, b h j d -> b h i d', attn, v), 'b h n d -> b n (h d)')
        return self.to_out(out)

class DecoupledMlaRopeAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.w_dq = nn.Linear(args.dim, args.q_compressed_dim, bias=False)
        self.q_norm = T5LayerNorm(args.q_compressed_dim)

        self.w_uq_qr = nn.Linear(args.q_compressed_dim, args.n_heads * (args.q_nope_head_dim + args.q_rope_head_dim),
                                 bias=False)
        self.rope_q = RotaryPositionalEmbedding(args, args.q_rope_head_dim)
        self.rope_k = RotaryPositionalEmbedding(args, args.k_rope_head_dim)

        self.w_dkv_kr = nn.Linear(args.dim, args.kv_compressed_dim + args.k_rope_head_dim, bias=False)
        self.kv_norm = T5LayerNorm(args.kv_compressed_dim)

        self.w_uk_uv = nn.Linear(args.kv_compressed_dim, args.n_kv_heads * (args.k_nope_head_dim + args.v_head_dim),
                                 bias=False)
        self.w_o = nn.Linear(args.n_heads * args.v_head_dim, args.dim, bias=False)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None, causal: bool = False, **kwargs) -> Tensor:
        batch_size, q_seq_len, _ = x.shape
        position_ids = torch.arange(q_seq_len, device=x.device).unsqueeze(0)

        # 1. Compressione
        compressed_q = self.q_norm(self.w_dq(x))
        compressed_kv_k_rope = self.w_dkv_kr(x)
        compressed_kv, k_rope = compressed_kv_k_rope.split([self.args.kv_compressed_dim, self.args.k_rope_head_dim],
                                                           dim=-1)
        compressed_kv = self.kv_norm(compressed_kv)

        # 2. Preparazione Query (Q)
        q_nope_q_rope = self.w_uq_qr(compressed_q)
        q_nope, q_rope = q_nope_q_rope.split(
            [self.args.n_heads * self.args.q_nope_head_dim, self.args.n_heads * self.args.q_rope_head_dim], dim=-1)
        q_nope = q_nope.view(batch_size, q_seq_len, self.args.n_heads, self.args.q_nope_head_dim).transpose(1, 2)
        q_rope = q_rope.view(batch_size, q_seq_len, self.args.n_heads, self.args.q_rope_head_dim).transpose(1, 2)
        q_rope = self.rope_q(q_rope, position_ids)
        query_states = torch.cat((q_nope, q_rope), dim=-1)

        # 3. Preparazione Key (K)
        k_rope = k_rope.view(batch_size, q_seq_len, 1, self.args.k_rope_head_dim).transpose(1, 2)
        k_rope = self.rope_k(k_rope, position_ids)

        k_nope_v_states = self.w_uk_uv(compressed_kv)
        k_nope, v_states = k_nope_v_states.split(
            [self.args.n_kv_heads * self.args.k_nope_head_dim, self.args.n_kv_heads * self.args.v_head_dim], dim=-1)

        k_nope = k_nope.view(batch_size, q_seq_len, self.args.n_kv_heads, self.args.k_nope_head_dim).transpose(1, 2)
        k_rope_expanded = repeat_kv_heads(k_rope, self.args.n_kv_heads)
        k_states = torch.cat((k_nope, k_rope_expanded), dim=-1)
        k_states = repeat_kv_heads(k_states, self.args.gqa_factor)

        # 4. Preparazione Value (V)
        v_states = v_states.view(batch_size, q_seq_len, self.args.n_kv_heads, self.args.v_head_dim).transpose(1, 2)
        v_states = repeat_kv_heads(v_states, self.args.gqa_factor)

        # 5. Calcolo Attenzione
        attn_mask = None
        if mask is not None or causal:
            attn_mask = torch.zeros(batch_size, 1, q_seq_len, q_seq_len, device=x.device, dtype=query_states.dtype)
            if mask is not None:
                attn_mask = attn_mask.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'))
            if causal:
                causal_mask = torch.triu(torch.ones((q_seq_len, q_seq_len), device=x.device, dtype=torch.bool),
                                         diagonal=1)
                attn_mask = attn_mask.masked_fill(causal_mask, float('-inf'))

        attn_output = torch.nn.functional.scaled_dot_product_attention(query_states, k_states, v_states,
                                                                       attn_mask=attn_mask)

        # 6. Proiezione Output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, q_seq_len,
                                                                    self.args.n_heads * self.args.v_head_dim)
        return self.w_o(attn_output)
# --- BLOCCHI ENCODER E DECODER ---
class EncoderBlock(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout, attention_args: ModelArgs, attention_type: AttentionType):
        super().__init__()
        if attention_type == AttentionType.RELATIVE_BIAS:
            self.attn = T5Attention(d_model, heads, causal=False, use_relative_bias=True, dropout=dropout)
        elif attention_type == AttentionType.MLA_ROPE_DECOUPLED:
            self.attn = DecoupledMlaRopeAttention(attention_args)
        self.ffn, self.norm1, self.norm2 = FeedForward(d_model, d_ff, dropout), T5LayerNorm(d_model), T5LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = x + self.dropout(self.attn(self.norm1(x), mask=mask, causal=False))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x

class DecoderBlock(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout, attention_args: ModelArgs, attention_type: AttentionType):
        super().__init__()
        # La self-attention usa la nuova logica
        if attention_type == AttentionType.RELATIVE_BIAS:
            self.self_attn = T5Attention(d_model, heads, causal=True, use_relative_bias=True, dropout=dropout)
        elif attention_type == AttentionType.MLA_ROPE_DECOUPLED:
            self.self_attn = DecoupledMlaRopeAttention(attention_args)
        # La cross-attention rimane sempre standard T5
        self.cross_attn = T5Attention(d_model, heads, causal=False, use_relative_bias=False, dropout=dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout=dropout)
        self.norm1, self.norm2, self.norm3 = T5LayerNorm(d_model), T5LayerNorm(d_model), T5LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context, src_mask, tgt_mask):
        x = x + self.dropout(self.self_attn(self.norm1(x), mask=tgt_mask, causal=True))
        x = x + self.dropout(self.cross_attn(self.norm2(x), context=context, context_mask=src_mask))
        x = x + self.dropout(self.ffn(self.norm3(x)))
        return x

# --- MODELLO PRINCIPALE ---
class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__(); self.embedding = nn.Embedding(vocab_size, d_model); self.scale = math.sqrt(d_model)
    def forward(self, x): return self.embedding(x) * self.scale

class Transformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, N: int, h: int, d_ff: int, dropout: float,
                 attention_args: ModelArgs, attention_type: AttentionType):
        super().__init__()
        self.embedding = InputEmbeddings(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.encoder_blocks = nn.ModuleList([EncoderBlock(d_model, h, d_ff, dropout, attention_args, attention_type) for _ in range(N)])
        self.decoder_blocks = nn.ModuleList([DecoderBlock(d_model, h, d_ff, dropout, attention_args, attention_type) for _ in range(N)])
        self.encoder_norm, self.decoder_norm = T5LayerNorm(d_model), T5LayerNorm(d_model)
        self.projection_layer = nn.Linear(d_model, vocab_size)

    def encode(self, src: torch.Tensor, src_padding_mask: torch.Tensor):
        x = self.dropout(self.embedding(src))
        for block in self.encoder_blocks: x = block(x, src_padding_mask)
        return self.encoder_norm(x)

    def decode(self, encoder_output: torch.Tensor, src_padding_mask: torch.Tensor, tgt: torch.Tensor, tgt_padding_mask: torch.Tensor):
        x = self.dropout(self.embedding(tgt))
        for block in self.decoder_blocks: x = block(x, encoder_output, src_padding_mask, tgt_padding_mask)
        return self.decoder_norm(x)

# --- FUNZIONE FACTORY (Ponte tra la tua config e il nuovo modello) ---
def build_transformer(vocab_size: int, seq_len: int, d_model: int, N: int, h: int, dropout: float, d_ff: int,
                      attention_type_str: str = 'mla_rope_decoupled', **kwargs) -> Transformer:

    attention_type = AttentionType(attention_type_str)
    attention_args = ModelArgs(dim=d_model, n_heads=h, max_position_embeddings=seq_len * 2)

    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None: nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding): nn.init.xavier_uniform_(module.weight)

    model = Transformer(
        vocab_size, d_model, N, h, d_ff, dropout, attention_args, attention_type
    )
    model.apply(_init_weights)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\n" + "="*60)
    print("--- Costruzione del Modello Transformer (per Confronto) ---")
    print(f"Architettura di Base: T5-Style (T5LayerNorm, FeedForward)")
    print(f"Meccanismo di Self-Attention: {attention_type.value}")
    print(f"Parametri totali (allenabili): {param_count:,}")
    print("="*60 + "\n")

    return model