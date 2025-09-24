from pathlib import Path


def get_config():
    return {
        "batch_size": 32,
        "num_epochs": 50,
        "lr": 1e-5,
        "seq_len": 512,
        "d_model": 512,
        "N": 4,
        "h": 8,
        "d_ff": 2048,
        "dropout": 0.1,
        "validate_every_n_epochs": 1, # Esegui la validazione ogni N epoche
        "num_validation_examples": -1, #feedback rapido durante lo sviluppo, impostare a -1 per usare l'intero set di validazione
        "model_folder": "weights",
        "model_basename": "nanosocrates_",
        "preload": None,
        "tokenizer_file": "../tokenizer/film_corpus_bpe_tokenizer.json",
        "corpus_file": "../dataset/training_corpus.txt",
    }
