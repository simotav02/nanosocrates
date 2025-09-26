# dataset_lib.py (MODIFICATO E DEFINITIVO)

import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer


# La funzione causal_mask non è più necessaria in questo file.
# Verrà gestita internamente dal modello o dalla logica di decoding.

class NanoSocratesDataset(Dataset):
    def __init__(self, raw_ds, tokenizer: Tokenizer, seq_len: int):
        super().__init__()
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.raw_ds = raw_ds

        self.pad_token_id = self.tokenizer.token_to_id("<PAD>")
        self.sot_token_id = self.tokenizer.token_to_id("<SOT>")
        self.eot_token_id = self.tokenizer.token_to_id("<EOT>")

        if self.pad_token_id is None or self.sot_token_id is None or self.eot_token_id is None:
            raise ValueError("Token <PAD>, <SOT> o <EOT> non trovati nel vocabolario del tokenizer.")

    def __len__(self):
        return len(self.raw_ds)

    def __getitem__(self, idx):
        src_tgt_pair = self.raw_ds[idx]
        src_text = src_tgt_pair['source']
        tgt_text = src_tgt_pair['target']

        enc_input_tokens = self.tokenizer.encode(src_text).ids
        dec_input_tokens = self.tokenizer.encode(tgt_text).ids

        if len(enc_input_tokens) > self.seq_len - 2:
            enc_input_tokens = enc_input_tokens[:self.seq_len - 2]
        if len(dec_input_tokens) > self.seq_len - 1:
            dec_input_tokens = dec_input_tokens[:self.seq_len - 1]

        enc_padding_needed = self.seq_len - len(enc_input_tokens) - 2
        dec_padding_needed = self.seq_len - len(dec_input_tokens) - 1

        encoder_input = torch.cat([
            torch.tensor([self.sot_token_id], dtype=torch.int64),
            torch.tensor(enc_input_tokens, dtype=torch.int64),
            torch.tensor([self.eot_token_id], dtype=torch.int64),
            torch.tensor([self.pad_token_id] * enc_padding_needed, dtype=torch.int64)
        ])

        decoder_input = torch.cat([
            torch.tensor([self.sot_token_id], dtype=torch.int64),
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            torch.tensor([self.pad_token_id] * dec_padding_needed, dtype=torch.int64)
        ])

        label = torch.cat([
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            torch.tensor([self.eot_token_id], dtype=torch.int64),
            torch.tensor([self.pad_token_id] * dec_padding_needed, dtype=torch.int64)
        ])

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            # --- MODIFICA CHIAVE ---
            # Generiamo maschere di padding semplici e booleane come richiesto da PyTorch.
            # Saranno di shape (seq_len,). Il DataLoader le raggrupperà in (batch_size, seq_len).
            # True indica una posizione da mascherare (ignorare).
            "encoder_mask": (encoder_input == self.pad_token_id),
            "decoder_mask": (decoder_input == self.pad_token_id),
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text,
        }