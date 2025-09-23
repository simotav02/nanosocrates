# ======================================================================================
# SEZIONE 2: CONFIGURAZIONE (config.py)
# ======================================================================================
from pathlib import Path


def get_config():
    # Iperparametri basati sugli HINTS della traccia
    return {
        "batch_size": 32,
        "num_epochs": 10,
        "lr": 5e-5,
        "seq_len": 256,  # HINT: Limit the maximum sequence length (256 or 512)
        "d_model": 256,  # HINT: Use a small hidden dimension (256 or 512)
        "N": 3,  # HINT: A model with 2-4 encoder/decoder layers
        "h": 4,  # HINT: A reduced number of attention heads (4 or 8)
        "d_ff": 1024,  # Feed-forward hidden dimension
        "dropout": 0.1,
        "model_folder": "weights",
        "model_basename": "nanosocrates_",
        "preload": None,
        "tokenizer_file": "../tokenizer/nanosocrates_hf_tokenizer.json",
        "corpus_file": "../dataset/training_corpus.txt",
    }