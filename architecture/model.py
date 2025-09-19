import torch
import torch.nn as nn
import math
from torch.utils.data import Dataset, DataLoader, random_split
from tokenizers import Tokenizer
from pathlib import Path
from tqdm import tqdm
import warnings


# ======================================================================================
# STEP 4.1: ARCHITETTURA DEL MODELLO (dal codice del professore, adattato)
# Questa sezione definisce tutti i blocchi costitutivi del Transformer.
# Non sono necessarie modifiche qui, poiché l'architettura è già generica.
# ======================================================================================

class LayerNormalization(nn.Module):
    def __init__(self, features: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


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
        self.d_model = d_model
        self.seq_len = seq_len
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


class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h"
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        return self.w_o(x)


class EncoderBlock(nn.Module):
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock,
                 cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock,
                 dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output,
                                                                                 src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x


class Decoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        return self.proj(x)


class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings,
                 src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        return self.projection_layer(x)


def build_transformer(vocab_size: int, seq_len: int, d_model: int = 256, N: int = 2, h: int = 4, dropout: float = 0.1,
                      d_ff: int = 1024) -> Transformer:
    # Creazione degli embedding layers (condivisi)
    embedding = InputEmbeddings(d_model, vocab_size)

    # Creazione dei positional encoding layers (condivisi)
    positional_encoding = PositionalEncoding(d_model, seq_len, dropout)

    # Creazione dei blocchi Encoder
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Creazione dei blocchi Decoder
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block,
                                     feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # Creazione di Encoder e Decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    # Projection layer
    projection_layer = ProjectionLayer(d_model, vocab_size)

    # Creazione del Transformer
    transformer = Transformer(encoder, decoder, embedding, embedding, positional_encoding, positional_encoding,
                              projection_layer)

    # Inizializzazione dei pesi
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer


# ======================================================================================
# STEP 4.2: CONFIGURAZIONE DEL MODELLO
# Qui impostiamo tutti gli iperparametri seguendo gli HINTS della traccia.
# ======================================================================================

def get_config():
    return {
        "batch_size": 16,
        "num_epochs": 10,  # Inizia con poche epoche per il debug
        "lr": 1e-4,
        "seq_len": 256,  # HINT: Limita la lunghezza della sequenza
        "d_model": 256,  # HINT: Usa una dimensione nascosta piccola
        "d_ff": 1024,  # Dimensione del feed-forward (solitamente 4 * d_model)
        "N": 3,  # HINT: Usa 2-4 layers per encoder/decoder
        "h": 4,  # HINT: Usa 4-8 attention heads
        "dropout": 0.1,
        "model_folder": "weights",
        "model_basename": "nanosocrates_",
        "preload": None,  # "latest" per caricare l'ultimo modello salvato
        "tokenizer_file": "nanosocrates_tokenizer.json",  # Il tuo tokenizer
        "corpus_file": "training_corpus.txt"  # Il tuo corpus
    }


# ======================================================================================
# STEP 4.3: CREAZIONE DEL DATASET PERSONALIZZATO
# Questa classe `NanoSocratesDataset` è progettata per leggere il tuo file
# `training_corpus.txt` e preparare i dati per il modello.
# ======================================================================================

def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0


class NanoSocratesDataset(Dataset):
    def __init__(self, corpus_path, tokenizer, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.tokenizer = tokenizer

        # Carica il corpus in memoria
        with open(corpus_path, 'r', encoding='utf-8') as f:
            self.lines = f.readlines()

        # Ottieni gli ID dei token speciali dal tuo tokenizer
        self.sot_token_id = self.tokenizer.token_to_id("<SOT>")
        self.eot_token_id = self.tokenizer.token_to_id("<EOT>")
        self.pad_token_id = self.tokenizer.token_to_id("<PAD>")

        # Controlla che i token speciali esistano
        if any(token_id is None for token_id in [self.sot_token_id, self.eot_token_id, self.pad_token_id]):
            raise ValueError("Uno o più token speciali (<SOT>, <EOT>, <PAD>) non trovati nel tokenizer.")

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx].strip()
        if '\t' not in line:
            # Salta righe vuote o malformate
            # In alternativa, puoi gestirle in modo più robusto
            print(f"Attenzione: riga {idx + 1} malformata, la salto.")
            return self.__getitem__((idx + 1) % len(self))  # Ritorna il prossimo elemento

        src_text, tgt_text = line.split('\t', 1)

        # Tokenizza l'input e l'output
        enc_input_tokens = self.tokenizer.encode(src_text).ids
        dec_input_tokens = self.tokenizer.encode(tgt_text).ids

        # Calcola il padding necessario
        # -1 per il token <EOT> aggiunto alla fine di enc_input
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 1
        # -1 per il token <SOT> all'inizio e -1 per <EOT> alla fine di dec_input
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        # Tronca le sequenze se sono troppo lunghe
        if enc_num_padding_tokens < 0:
            enc_input_tokens = enc_input_tokens[:self.seq_len - 1]
            enc_num_padding_tokens = 0
        if dec_num_padding_tokens < 0:
            dec_input_tokens = dec_input_tokens[:self.seq_len - 1]
            dec_num_padding_tokens = 0

        # Crea i tensori per l'encoder e il decoder
        encoder_input = torch.cat([
            torch.tensor(enc_input_tokens, dtype=torch.int64),
            torch.tensor([self.eot_token_id], dtype=torch.int64),  # Aggiungiamo <EOT> per segnalare la fine dell'input
            torch.tensor([self.pad_token_id] * enc_num_padding_tokens, dtype=torch.int64),
        ])

        # L'input del decoder inizia con <SOT>
        decoder_input = torch.cat([
            torch.tensor([self.sot_token_id], dtype=torch.int64),
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            torch.tensor([self.pad_token_id] * dec_num_padding_tokens, dtype=torch.int64),
        ])

        # L'etichetta (label) finisce con <EOT>
        label = torch.cat([
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            torch.tensor([self.eot_token_id], dtype=torch.int64),
            torch.tensor([self.pad_token_id] * dec_num_padding_tokens, dtype=torch.int64),
        ])

        # Assicurati che le dimensioni siano corrette
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input != self.pad_token_id).unsqueeze(0).unsqueeze(0).int(),  # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token_id).unsqueeze(0).int() & causal_mask(
                decoder_input.size(0)),  # (1, seq_len) & (1, seq_len, seq_len)
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text,
        }


# ======================================================================================
# STEP 4.4: LOGICA DI TRAINING
# Questa è la funzione principale che orchestra la creazione del modello,
# il caricamento dei dati e il ciclo di addestramento.
# ======================================================================================

def get_ds(config):
    # Carica il tokenizer custom
    tokenizer = Tokenizer.from_file(config['tokenizer_file'])

    # Crea il dataset
    dataset = NanoSocratesDataset(config['corpus_file'], tokenizer, config['seq_len'])

    # Suddividi in training e validation set (90% training, 10% validation)
    train_ds_size = int(0.9 * len(dataset))
    val_ds_size = len(dataset) - train_ds_size
    train_ds, val_ds = random_split(dataset, [train_ds_size, val_ds_size])

    # Crea i DataLoader
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)  # Batch size 1 per la validazione

    return train_dataloader, val_dataloader, tokenizer


def train_model(config):
    # Seleziona il device (GPU se disponibile)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    # Carica dataloader e tokenizer
    train_dataloader, val_dataloader, tokenizer = get_ds(config)

    # Costruisci il modello
    model = build_transformer(
        vocab_size=tokenizer.get_vocab_size(),
        seq_len=config['seq_len'],
        d_model=config['d_model'],
        N=config['N'],
        h=config['h'],
        dropout=config['dropout'],
        d_ff=config['d_ff']
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], eps=1e-9)

    initial_epoch = 0

    # Gestione del preload (se vuoi continuare un training)
    # ... (omesso per semplicità, ma puoi aggiungerlo dal codice originale)

    # La loss function ignora il padding token
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id('<PAD>'), label_smoothing=0.1).to(device)

    # Ciclo di addestramento
    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")

        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)

            # Esegui i tensori attraverso il modello
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)

            label = batch['label'].to(device)

            # Calcola la loss
            loss = loss_fn(proj_output.view(-1, tokenizer.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # Salva il modello alla fine di ogni epoca
        model_filename = f"{config['model_folder']}/{config['model_basename']}{epoch:02d}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, model_filename)

        # TODO: Esegui la validazione (run_validation) alla fine di ogni epoca.
        # La logica di validazione (greedy_decode, etc.) dal codice originale
        # può essere adattata qui per misurare le performance sul validation set.


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)