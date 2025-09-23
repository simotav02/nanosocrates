from pathlib import Path


def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 3e-5,
        "seq_len": 256,
        "d_model": 256,
        "N": 2,
        "h": 4,
        "d_ff": 1024,
        "dropout": 0.1,
        "validate_every_n_epochs": 1, # Esegui la validazione ogni N epoche
        "num_validation_examples": -1, #feedback rapido durante lo sviluppo, impostare a -1 per usare l'intero set di validazione
        "model_folder": "weights",
        "model_basename": "nanosocrates_",
        "preload": None,
        "tokenizer_file": "../tokenizer/nanosocrates_hf_tokenizer.json",
        "corpus_file": "../dataset/training_corpus.txt",
    }
