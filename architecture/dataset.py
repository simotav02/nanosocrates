# ======================================================================================
# SEZIONE 3: GESTIONE DATASET (dataset.py)
# ======================================================================================
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0


class NanoSocratesDataset(Dataset):
    def __init__(self, corpus_path, tokenizer, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.tokenizer = tokenizer

        with open(corpus_path, 'r', encoding='utf-8') as f:
            self.lines = f.readlines()

        self.sot_token = torch.tensor([tokenizer.token_to_id("<SOT>")], dtype=torch.int64)
        self.eot_token = torch.tensor([tokenizer.token_to_id("<EOT>")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer.token_to_id("<PAD>")], dtype=torch.int64)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx].strip()
        if '\t' not in line:
            return self.__getitem__((idx + 1) % len(self))

        src_text, tgt_text = line.split('\t', 1)

        enc_input_tokens = self.tokenizer.encode(src_text).ids
        dec_input_tokens = self.tokenizer.encode(tgt_text).ids

        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 1
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        if enc_num_padding_tokens < 0:
            enc_input_tokens = enc_input_tokens[:self.seq_len - 1]
            enc_num_padding_tokens = 0
        if dec_num_padding_tokens < 0:
            dec_input_tokens = dec_input_tokens[:self.seq_len - 1]
            dec_num_padding_tokens = 0

        encoder_input = torch.cat([
            torch.tensor(enc_input_tokens, dtype=torch.int64),
            self.eot_token,
            torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),
        ], dim=0)

        decoder_input = torch.cat([
            self.sot_token,
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
        ], dim=0)

        label = torch.cat([
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            self.eot_token,
            torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
        ], dim=0)

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),  # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            # (1, seq_len) & (1, seq_len, seq_len)
            "label": label,  # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }