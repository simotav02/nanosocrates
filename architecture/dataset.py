import torch
from torch.utils.data import Dataset
from vecchio.tokenizer.tokenizer import NanoSocratesTokenizer


def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0


class NanoSocratesDataset(Dataset):
    def __init__(self, corpus_path, tokenizer: NanoSocratesTokenizer, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.tokenizer = tokenizer

        with open(corpus_path, 'r', encoding='utf-8') as f:
            self.lines = f.readlines()

        # Accediamo agli ID tramite il dizionario .vocab
        self.sot_token_id = self.tokenizer.vocab["<SOT>"]
        self.eot_token_id = self.tokenizer.vocab["<EOT>"]
        self.pad_token_id = self.tokenizer.vocab["<PAD>"]

        # Creiamo tensori una sola volta per efficienza, non serve per il PAD
        self.sot_token = torch.tensor([self.sot_token_id], dtype=torch.int64)
        self.eot_token = torch.tensor([self.eot_token_id], dtype=torch.int64)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx].strip()
        if '\t' not in line:
            return self.__getitem__((idx + 1) % len(self))

        src_text, tgt_text = line.split('\t', 1)

        # Usiamo il tuo metodo .encode() che restituisce già una lista di ID VEDERE SE LO POSSIAMO ADATTARE TIPO HUGGING FACES
        enc_input_tokens = self.tokenizer.encode(src_text)
        dec_input_tokens = self.tokenizer.encode(tgt_text)

        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        # Make sure the number of padding tokens is not negative. If it is, the sentence is too long
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        # # Stiamo troncando le frasi troppo lunghe perdendo parte dell'informazione
        # if enc_num_padding_tokens < 0:
        #     enc_input_tokens = enc_input_tokens[:self.seq_len - 2]
        #     enc_num_padding_tokens = 0
        # if dec_num_padding_tokens < 0:
        #     dec_input_tokens = dec_input_tokens[:self.seq_len - 1]
        #     dec_num_padding_tokens = 0

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
            self.eot_token,
            torch.tensor([self.pad_token_id] * dec_num_padding_tokens, dtype=torch.int64),
        ], dim=0)

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