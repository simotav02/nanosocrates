# dataset_lib.py

import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer


class NanoSocratesDataset(Dataset):
    def __init__(self, raw_ds, tokenizer: Tokenizer, seq_len: int):
        super().__init__()
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.raw_ds = raw_ds

        self.pad_token_id = self.tokenizer.token_to_id("<PAD>")
        self.sot_token_id = self.tokenizer.token_to_id("<SOT>")
        self.eot_token_id = self.tokenizer.token_to_id("<EOT>")

        # --- NUOVA PARTE ---
        # Otteniamo gli ID di tutti i token <extra_id_...>
        self.extra_id_token_ids = {
            self.tokenizer.token_to_id(f"<extra_id_{i}>") for i in range(150)
        }
        # Rimuoviamo eventuali None se alcuni token non esistono (buona pratica)
        self.extra_id_token_ids.discard(None)
        # --- FINE NUOVA PARTE ---

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

        # ---- LA LOGICA QUI SOTTO È GIÀ CORRETTA PER T5 ----
        # decoder_input: Inizia con <SOT> ed è il target shiftato a destra
        # label: È il target originale, termina con <EOT>

        enc_padding_needed = self.seq_len - len(enc_input_tokens) - 2
        dec_padding_needed = self.seq_len - len(dec_input_tokens) - 1  # Per il decoder_input
        label_padding_needed = self.seq_len - len(dec_input_tokens) - 1  # Per la label

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

        label_tokens = torch.cat([
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            torch.tensor([self.eot_token_id], dtype=torch.int64),
            torch.tensor([self.pad_token_id] * label_padding_needed, dtype=torch.int64)
        ])

        # --- MODIFICA CRUCIALE ---
        # Mascheriamo i token <extra_id_...> nella label in modo che la loss li ignori.
        # Sostituiamo il loro ID con il pad_token_id.
        #for extra_id in self.extra_id_token_ids:
        #    label_tokens[label_tokens == extra_id] = self.pad_token_id
        # --- FINE MODIFICA ---

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label_tokens.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input == self.pad_token_id),
            "decoder_mask": (decoder_input == self.pad_token_id),
            "label": label_tokens,  # Usiamo la label modificata
            "src_text": src_text,
            "tgt_text": tgt_text,
        }