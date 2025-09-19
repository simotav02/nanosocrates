# ======================================================================================
# SEZIONE 2: CONFIGURAZIONE (config.py)
# ======================================================================================
from pathlib import Path


def get_config():
    # Iperparametri basati sugli HINTS della traccia
    return {
        "batch_size": 16,
        "num_epochs": 10,
        "lr": 1e-4,
        "seq_len": 512,  # HINT: Limit the maximum sequence length (256 or 512)
        "d_model": 512,  # HINT: Use a small hidden dimension (256 or 512)
        "N": 4,  # HINT: A model with 2-4 encoder/decoder layers
        "h": 8,  # HINT: A reduced number of attention heads (4 or 8)
        "d_ff": 1024,  # Feed-forward hidden dimension
        "dropout": 0.1,
        "model_folder": "weights",
        "model_basename": "nanosocrates_",
        "preload": None,
        "tokenizer_file": "/Users/simonetavilla/Documents/nanosocrates/tokenizer/nanosocrates_tokenizer.json",
        "corpus_file": "../dataset/training_corpus.txt",
    }