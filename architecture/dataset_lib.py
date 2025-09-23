import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer


def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0


class NanoSocratesDataset(Dataset):
    def __init__(self, corpus_path: str, tokenizer: Tokenizer, seq_len: int):
        super().__init__()
        self.seq_len = seq_len
        self.tokenizer = tokenizer

        with open(corpus_path, 'r', encoding='utf-8') as f:
            self.lines = f.readlines()

        # Restituisce None se il token non esiste, quindi possiamo gestirlo se necessario
        self.sot_token_id = self.tokenizer.token_to_id("<SOT>")
        self.eot_token_id = self.tokenizer.token_to_id("<EOT>")
        self.pad_token_id = self.tokenizer.token_to_id("<PAD>")

        # Controlliamo che i token speciali siano stati trovati
        if any(id is None for id in [self.sot_token_id, self.eot_token_id, self.pad_token_id]):
            raise ValueError("Uno o più token speciali (<SOT>, <EOT>, <PAD>) non trovati nel tokenizer.")

        # Creiamo tensori una sola volta per efficienza
        self.sot_token = torch.tensor([self.sot_token_id], dtype=torch.int64)
        self.eot_token = torch.tensor([self.eot_token_id], dtype=torch.int64)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx].strip()
        if '\t' not in line:
            import random
            return self.__getitem__(random.randint(0, len(self) - 1))

        src_text, tgt_text = line.split('\t', 1)

        enc_input_tokens = self.tokenizer.encode(src_text).ids
        dec_input_tokens = self.tokenizer.encode(tgt_text).ids

        # Il resto della logica per il padding rimane quasi identico
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  # SOT e EOT
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1  # SOT

        # Gestiamo il caso di frasi troppo lunghe troncandole
        if enc_num_padding_tokens < 0:
            # Tronchiamo l'input dell'encoder
            enc_input_tokens = enc_input_tokens[:self.seq_len - 2]
            enc_num_padding_tokens = 0

        if dec_num_padding_tokens < 0:
            # Tronchiamo l'input del decoder e il label corrispondente
            dec_input_tokens = dec_input_tokens[:self.seq_len - 1]
            dec_num_padding_tokens = 0

        # Costruiamo i tensori
        encoder_input = torch.cat([
            self.sot_token,
            torch.tensor(enc_input_tokens, dtype=torch.int64),
            self.eot_token,
            torch.tensor([self.pad_token_id] * enc_num_padding_tokens, dtype=torch.int64),
        ], dim=0)

        decoder_input = torch.cat([
            self.sot_token,
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            torch.tensor([self.pad_token_id] * dec_num_padding_tokens, dtype=torch.int64),
        ], dim=0)

        label = torch.cat([
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            self.eot_token,  # L'obiettivo è predire EOT
            torch.tensor([self.pad_token_id] * dec_num_padding_tokens, dtype=torch.int64),
        ], dim=0)

        # Assicuriamoci che le dimensioni siano corrette
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input != self.pad_token_id).unsqueeze(0).unsqueeze(0).int(),
            "decoder_mask": (decoder_input != self.pad_token_id).unsqueeze(0).int() & causal_mask(
                decoder_input.size(0)),
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text,
        }