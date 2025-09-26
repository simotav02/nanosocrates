# model.py (VERSIONE FINALE EFFICIENTE CON CORREZIONE DEL TIPO DI DATO)

import torch
import torch.nn as nn
import math


class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, N: int, h: int, d_ff: int, seq_len: int, dropout: float):
        super().__init__()
        self.src_embed = InputEmbeddings(d_model, vocab_size)
        self.tgt_embed = InputEmbeddings(d_model, vocab_size)
        self.positional_encoding = PositionalEncoding(d_model, seq_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=h, dim_feedforward=d_ff, dropout=dropout,
            activation='relu', norm_first=True, batch_first=True
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=h, dim_feedforward=d_ff, dropout=dropout,
            activation='relu', norm_first=True, batch_first=True
        )

        encoder_norm = nn.LayerNorm(d_model)
        decoder_norm = nn.LayerNorm(d_model)

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=N, norm=encoder_norm)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=N, norm=decoder_norm)
        self.projection_layer = nn.Linear(d_model, vocab_size)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        src_emb = self.src_embed(src)
        src_pos = self.positional_encoding(src_emb)
        src_padding_mask = (src_mask == 0).squeeze(1).squeeze(1)
        return self.encoder(src_pos, src_key_padding_mask=src_padding_mask)

    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor,
               tgt_mask: torch.Tensor) -> torch.Tensor:
        tgt_emb = self.tgt_embed(tgt)
        tgt_pos = self.positional_encoding(tgt_emb)

        src_padding_mask = (src_mask == 0).squeeze(1).squeeze(1)

        # --- MODIFICA CHIAVE QUI ---
        # Prima di usare masked_fill_ con valori float, convertiamo la maschera a float.
        # .clone() per non modificare il tensore originale.
        # .float() per cambiare il tipo da int a float.
        tgt_causal_mask = tgt_mask.clone().float()  # <--- AGGIUNTO .float()

        # Ora questa operazione funzionerà perché tgt_causal_mask è di tipo float
        tgt_causal_mask.masked_fill_(tgt_causal_mask == 0, float('-inf')).masked_fill_(tgt_causal_mask == 1, float(0.0))

        return self.decoder(
            tgt=tgt_pos,
            memory=encoder_output,
            tgt_mask=tgt_causal_mask,
            memory_key_padding_mask=src_padding_mask
        )

    def project(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection_layer(x)


def build_transformer(vocab_size: int, seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1,
                      d_ff: int = 2048) -> Transformer:
    model = Transformer(
        vocab_size=vocab_size, d_model=d_model, N=N, h=h, d_ff=d_ff,
        seq_len=seq_len, dropout=dropout
    )
    print(
        f"Modello Transformer (stile T5, efficiente) costruito con {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parametri.")
    return model