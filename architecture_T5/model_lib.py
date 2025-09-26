# model.py (VERSIONE AVANZATA CON T5 RELATIVE POSITIONAL BIAS)

import torch
from torch import nn
import torch.nn.functional as F
import math
from einops import rearrange


# --- MODULO 1: Codice per i T5 Relative Positional Bias ---
# Questo codice è adattato da quello che hai fornito.

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
        return qk_dots + (bias * self.scale)


# --- MODULO 2: Componenti Base del Transformer ---

class T5LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


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
    def __init__(self, d_model, heads, causal=False, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.causal = causal

        self.multi_head_attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.relative_position_bias = T5RelativePositionBias(
            scale=(d_model // heads) ** -0.5, causal=causal, heads=heads
        )

    def forward(self, x, context=None, mask=None, context_mask=None):
        # nn.MultiheadAttention aspetta una maschera di padding booleana
        # e una maschera di attention additiva
        key_padding_mask = mask
        if context is not None:
            key_padding_mask = context_mask

        attn_mask = None
        if self.causal:
            attn_mask = nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device)

        # Calcolo dell'attenzione
        if context is None:
            context = x  # Self-attention

        # PyTorch < 2.0.0 non ha `attn_mask` nel forward, lo aggiungiamo a mano
        # Qui usiamo un trucco per aggiungere il bias relativo
        q = self.multi_head_attn.in_proj_q(x)
        k = self.multi_head_attn.in_proj_k(context)
        v = self.multi_head_attn.in_proj_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))

        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k)

        # Aggiungiamo il bias relativo
        sim_with_bias = self.relative_position_bias(sim)

        if attn_mask is not None:
            sim_with_bias += attn_mask

        if key_padding_mask is not None:
            sim_with_bias = sim_with_bias.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attn = sim_with_bias.softmax(dim=-1)
        attn = F.dropout(attn, p=self.multi_head_attn.dropout, training=self.training)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.multi_head_attn.out_proj(out)


# --- MODULO 3: Blocchi Encoder e Decoder Custom ---

class EncoderBlock(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super().__init__()
        self.attn = T5Attention(d_model, heads, causal=False, dropout=dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout=dropout)
        self.norm1 = T5LayerNorm(d_model)
        self.norm2 = T5LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        # Pre-LN
        norm_x = self.norm1(x)
        x = x + self.dropout1(self.attn(norm_x, mask=mask))

        norm_x = self.norm2(x)
        x = x + self.dropout2(self.ffn(norm_x))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super().__init__()
        self.self_attn = T5Attention(d_model, heads, causal=True, dropout=dropout)
        self.cross_attn = T5Attention(d_model, heads, causal=False, dropout=dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout=dropout)
        self.norm1 = T5LayerNorm(d_model)
        self.norm2 = T5LayerNorm(d_model)
        self.norm3 = T5LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, context, src_mask, tgt_mask):
        # Self-attention con Pre-LN
        norm_x = self.norm1(x)
        x = x + self.dropout1(self.self_attn(norm_x, mask=tgt_mask))

        # Cross-attention con Pre-LN
        norm_x = self.norm2(x)
        x = x + self.dropout2(self.cross_attn(norm_x, context=context, context_mask=src_mask))

        # Feed-forward con Pre-LN
        norm_x = self.norm3(x)
        x = x + self.dropout3(self.ffn(norm_x))
        return x


# --- MODULO 4: Modello Transformer Completo ---

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.scale = math.sqrt(d_model)

    def forward(self, x):
        return self.embedding(x) * self.scale


class Transformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, N: int, h: int, d_ff: int, dropout: float):
        super().__init__()
        self.embedding = InputEmbeddings(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

        self.encoder_blocks = nn.ModuleList([EncoderBlock(d_model, h, d_ff, dropout) for _ in range(N)])
        self.decoder_blocks = nn.ModuleList([DecoderBlock(d_model, h, d_ff, dropout) for _ in range(N)])

        self.encoder_norm = T5LayerNorm(d_model)
        self.decoder_norm = T5LayerNorm(d_model)

        self.projection_layer = nn.Linear(d_model, vocab_size)
        self.embedding.embedding.weight = self.projection_layer.weight  # Weight tying

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

    def project(self, x: torch.Tensor):
        return self.projection_layer(x)


def build_transformer(vocab_size: int, seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1,
                      d_ff: int = 2048) -> Transformer:
    # Nota: seq_len non è più usato direttamente nel modello, ma lo manteniamo per compatibilità con l'interfaccia
    model = Transformer(
        vocab_size=vocab_size, d_model=d_model, N=N, h=h, d_ff=d_ff, dropout=dropout
    )
    print(
        f"Modello Transformer (con T5 Relative Bias) costruito con {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parametri.")
    return model