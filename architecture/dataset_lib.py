import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
import os


def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0


class NanoSocratesDataset(Dataset):

    def __init__(self, data_dir: str, tokenizer: Tokenizer, seq_len: int, split: str = "train"):
        super().__init__()
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.split = split

        source_path = os.path.join(data_dir, f"{split}.source")
        target_path = os.path.join(data_dir, f"{split}.target")

        # Carichiamo le righe da entrambi i file
        print(f"Caricamento del dataset '{split}'...")
        with open(source_path, 'r', encoding='utf-8') as f:
            self.source_lines = f.readlines()
        with open(target_path, 'r', encoding='utf-8') as f:
            self.target_lines = f.readlines()

        # Controllo di coerenza: i file devono avere lo stesso numero di righe
        assert len(self.source_lines) == len(self.target_lines), \
            "I file source e target non hanno lo stesso numero di righe!"

        # Otteniamo gli ID dei token speciali una sola volta
        self.pad_token_id = self.tokenizer.token_to_id("<PAD>")
        self.sot_token_id = self.tokenizer.token_to_id("<SOT>")  # Start of Text
        self.eot_token_id = self.tokenizer.token_to_id("<EOT>")  # End of Text

        if self.pad_token_id is None or self.sot_token_id is None or self.eot_token_id is None:
            raise ValueError("Token <PAD>, <SOT> o <EOT> non trovati nel vocabolario del tokenizer.")

    def __len__(self):
        return len(self.source_lines)

    def __getitem__(self, idx):
        src_text = self.source_lines[idx].strip()
        tgt_text = self.target_lines[idx].strip()

        # Tokenizziamo le sequenze di input e di target
        enc_input_tokens = self.tokenizer.encode(src_text).ids
        dec_input_tokens = self.tokenizer.encode(tgt_text).ids

        # Calcoliamo quanti token di padding sono necessari
        # Per l'encoder: [SOT] + source + [EOT]
        enc_padding_needed = self.seq_len - len(enc_input_tokens) - 2
        # Per il decoder: [SOT] + target
        dec_padding_needed = self.seq_len - len(dec_input_tokens) - 1

        # Gestiamo sequenze troppo lunghe (troncamento)
        if enc_padding_needed < 0:
            enc_input_tokens = enc_input_tokens[:self.seq_len - 2]
            enc_padding_needed = 0

        if dec_padding_needed < 0:
            dec_input_tokens = dec_input_tokens[:self.seq_len - 1]
            dec_padding_needed = 0

        # Input per l'Encoder: [SOT] + source_tokens + [EOT] + [PAD]...
        encoder_input = torch.cat([
            torch.tensor([self.sot_token_id]),
            torch.tensor(enc_input_tokens, dtype=torch.int64),
            torch.tensor([self.eot_token_id]),
            torch.tensor([self.pad_token_id] * enc_padding_needed, dtype=torch.int64)
        ])

        # Input per il Decoder (teacher forcing): [SOT] + target_tokens + [PAD]...
        decoder_input = torch.cat([
            torch.tensor([self.sot_token_id]),
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            torch.tensor([self.pad_token_id] * dec_padding_needed, dtype=torch.int64)
        ])

        # Label (ciò che il decoder deve predire): target_tokens + [EOT] + [PAD]...
        label = torch.cat([
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            torch.tensor([self.eot_token_id]),
            torch.tensor([self.pad_token_id] * dec_padding_needed, dtype=torch.int64)
        ])

        # Assicuriamoci che tutti i tensori abbiano la lunghezza corretta
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            "encoder_mask": (encoder_input != self.pad_token_id).unsqueeze(0).unsqueeze(0).int(),  # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token_id).unsqueeze(0).int() & causal_mask(
                decoder_input.size(0)),  # (1, seq_len) & (1, seq_len, seq_len)
            "label": label,  # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }
