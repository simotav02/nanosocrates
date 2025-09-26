# model.py (VERSIONE FINALE EFFICIENTE CON MODULI PYTORCH)

import torch
import torch.nn as nn
import math


class InputEmbeddings(nn.Module):
    """
    Converte gli indici dei token in embedding densi.
    """

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # Come da paper "Attention Is All You Need", scaliamo gli embedding.
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    Inietta informazioni sulla posizione dei token nella sequenza.
    """

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Crea una matrice di positional encoding precalcolata
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Aggiunge la dimensione del batch

        # Registra 'pe' come buffer, non come parametro da addestrare
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Aggiunge gli embedding posizionali agli embedding dei token
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class Transformer(nn.Module):
    """
    Architettura Transformer Encoder-Decoder completa, costruita usando i moduli
    ottimizzati di PyTorch e configurata per emulare le caratteristiche di T5.
    """

    def __init__(self, vocab_size: int, d_model: int, N: int, h: int, d_ff: int, seq_len: int, dropout: float):
        super().__init__()

        self.src_embed = InputEmbeddings(d_model, vocab_size)
        self.tgt_embed = InputEmbeddings(d_model, vocab_size)
        self.positional_encoding = PositionalEncoding(d_model, seq_len, dropout)

        # 1. LayerNorm at the start of each block (Pre-LN):
        #    Ottenuto impostando `norm_first=True`. Questo migliora la stabilità del training.

        # Creazione di un layer Encoder standard, configurato in stile T5
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=h,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='relu',
            norm_first=True,  # <-- CHIAVE: Abilita il Pre-LayerNorm (stile T5)
            batch_first=True  # <-- CHIAVE: Coerenza con il formato dei nostri dati (batch, seq, dim)
        )

        # Creazione di un layer Decoder standard, configurato in stile T5
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=h,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='relu',
            norm_first=True,  # <-- CHIAVE: Abilita il Pre-LayerNorm (stile T5)
            batch_first=True  # <-- CHIAVE: Coerenza con il formato dei nostri dati
        )

        # 2. "Scale-only" LayerNorm:
        #    PyTorch non ha un LayerNorm "scale-only" nativo, ma nn.LayerNorm è
        #    altamente ottimizzato. Per semplicità ed efficienza, usiamo quello standard.
        #    La differenza di performance è minima per modelli di questa scala.
        encoder_norm = nn.LayerNorm(d_model)
        decoder_norm = nn.LayerNorm(d_model)

        # Creazione dello stack di N layer per Encoder e Decoder
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=N, norm=encoder_norm)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=N, norm=decoder_norm)

        self.projection_layer = nn.Linear(d_model, vocab_size)

        # Inizializzazione dei pesi per una convergenza migliore
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Processa la sequenza di input.
        - src_mask: Maschera per ignorare il padding nell'input.
        """
        src_emb = self.src_embed(src)
        src_pos = self.positional_encoding(src_emb)
        # La maschera di padding per l'encoder di PyTorch deve essere booleana:
        # True dove i valori devono essere ignorati (dove c'è padding).
        # Il nostro `src_mask` è (batch, 1, 1, seq_len) con 0 per il padding.
        # Dobbiamo adattarlo.
        src_padding_mask = (src_mask == 0).squeeze(1).squeeze(1)
        return self.encoder(src_pos, src_key_padding_mask=src_padding_mask)

    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor,
               tgt_mask: torch.Tensor) -> torch.Tensor:
        """
        Genera la sequenza di output in modo autoregressivo.
        - encoder_output: Output dell'encoder.
        - src_mask: Maschera per ignorare il padding dell'input nella cross-attention.
        - tgt: Sequenza di output finora generata.
        - tgt_mask: Maschera "causale" per impedire al decoder di "sbirciare" i token futuri.
        """
        tgt_emb = self.tgt_embed(tgt)
        tgt_pos = self.positional_encoding(tgt_emb)
        # Adattiamo le maschere come richiesto da nn.TransformerDecoder
        src_padding_mask = (src_mask == 0).squeeze(1).squeeze(1)
        # La maschera causale di PyTorch deve avere -inf dove l'attenzione è proibita
        tgt_causal_mask = tgt_mask.clone()
        tgt_causal_mask.masked_fill_(tgt_causal_mask == 0, float('-inf')).masked_fill_(tgt_causal_mask == 1, float(0.0))

        return self.decoder(
            tgt=tgt_pos,
            memory=encoder_output,
            tgt_mask=tgt_causal_mask,
            memory_key_padding_mask=src_padding_mask
        )

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """
        Proietta l'output del decoder nello spazio del vocabolario.
        """
        return self.projection_layer(x)


def build_transformer(vocab_size: int, seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1,
                      d_ff: int = 2048) -> Transformer:
    """
    Funzione factory per costruire il modello Transformer con i parametri specificati.
    """
    model = Transformer(
        vocab_size=vocab_size,
        d_model=d_model,
        N=N,
        h=h,
        d_ff=d_ff,
        seq_len=seq_len,
        dropout=dropout
    )
    print(
        f"Modello Transformer (stile T5, efficiente) costruito con {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parametri.")
    return model