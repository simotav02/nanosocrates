import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer


class NanoSocratesDataset(Dataset):
    """
    A PyTorch Dataset for handling the NanoSocrates dataset.
    It takes raw source-target text pairs, tokenizes them, and prepares them
    into a format suitable for training a sequence-to-sequence Transformer model.
    """

    def __init__(self, raw_ds, tokenizer: Tokenizer, seq_len: int):
        """
        Initializes the dataset.

        Args:
            raw_ds: The raw dataset containing source-target pairs.
            tokenizer: The tokenizer to be used for encoding text.
            seq_len: The fixed sequence length for all inputs and outputs.
        """
        super().__init__()
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.raw_ds = raw_ds

        # Retrieve special token IDs from the tokenizer's vocabulary.
        self.pad_token_id = self.tokenizer.token_to_id("<PAD>")
        self.sot_token_id = self.tokenizer.token_to_id("<SOT>")  # Start Of Text
        self.eot_token_id = self.tokenizer.token_to_id("<EOT>")  # End Of Text

        # A set of extra_id tokens, useful for tasks like T5's denoising objective, though not used in this getitem.
        self.extra_id_token_ids = {
            self.tokenizer.token_to_id(f"<extra_id_{i}>") for i in range(150)
        }
        self.extra_id_token_ids.discard(None)  # Remove None if a token doesn't exist.

        # Ensure that essential special tokens are present in the tokenizer.
        if self.pad_token_id is None or self.sot_token_id is None or self.eot_token_id is None:
            raise ValueError("Token <PAD>, <SOT> o <EOT> non trovati nel vocabolario del tokenizer.")

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.raw_ds)

    def __getitem__(self, idx):
        """
        Retrieves and processes a single data sample from the dataset.
        This involves tokenizing, truncating, and padding the source and target sequences.
        """
        src_tgt_pair = self.raw_ds[idx]
        src_text = src_tgt_pair['source']
        tgt_text = src_tgt_pair['target']

        # Tokenize the source and target texts.
        enc_input_tokens = self.tokenizer.encode(src_text).ids
        dec_input_tokens = self.tokenizer.encode(tgt_text).ids

        # Truncate sequences if they are too long to fit the <SOT> and <EOT> tokens within seq_len.
        # Encoder input needs space for <SOT> and <EOT> (seq_len - 2).
        if len(enc_input_tokens) > self.seq_len - 2:
            enc_input_tokens = enc_input_tokens[:self.seq_len - 2]
        # Decoder input/label needs space for <SOT> at the beginning or <EOT> at the end (seq_len - 1).
        if len(dec_input_tokens) > self.seq_len - 1:
            dec_input_tokens = dec_input_tokens[:self.seq_len - 1]

        # Calculate the number of padding tokens needed to reach seq_len.
        enc_padding_needed = self.seq_len - len(enc_input_tokens) - 2  # For encoder_input (<SOT> and <EOT>)
        dec_padding_needed = self.seq_len - len(dec_input_tokens) - 1  # For decoder_input (<SOT>)
        label_padding_needed = self.seq_len - len(dec_input_tokens) - 1  # For the label (<EOT>)

        # Construct the encoder input tensor.
        # Format: <SOT> source_tokens <EOT> <PAD>...<PAD>
        encoder_input = torch.cat([
            torch.tensor([self.sot_token_id], dtype=torch.int64),
            torch.tensor(enc_input_tokens, dtype=torch.int64),
            torch.tensor([self.eot_token_id], dtype=torch.int64),
            torch.tensor([self.pad_token_id] * enc_padding_needed, dtype=torch.int64)
        ])

        # Construct the decoder input tensor (for teacher forcing).
        # Format: <SOT> target_tokens <PAD>...<PAD>
        decoder_input = torch.cat([
            torch.tensor([self.sot_token_id], dtype=torch.int64),
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            torch.tensor([self.pad_token_id] * dec_padding_needed, dtype=torch.int64)
        ])

        # Construct the label tensor (what the model should predict).
        # Format: target_tokens <EOT> <PAD>...<PAD>
        label_tokens = torch.cat([
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            torch.tensor([self.eot_token_id], dtype=torch.int64),
            torch.tensor([self.pad_token_id] * label_padding_needed, dtype=torch.int64)
        ])

        # Ensure all tensors are padded/truncated to the fixed sequence length.
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label_tokens.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            "encoder_mask": (encoder_input == self.pad_token_id),  # (seq_len), True for padding tokens
            "decoder_mask": (decoder_input == self.pad_token_id),  # (seq_len), True for padding tokens
            "label": label_tokens,  # (seq_len)
            "src_text": src_text,  # Original source text
            "tgt_text": tgt_text,  # Original target text
        }
