# architecture/config.py

from pathlib import Path


def get_config():
    # Iperparametri basati sugli HINTS della traccia
    return {
        "batch_size": 32,
        "num_epochs": 50,
        "lr": 1e-5,
        "seq_len": 256,
        "d_model": 512,
        "N": 4,
        "h": 8,
        "d_ff": 2048,
        "dropout": 0.1,

        # <--- NUOVO: Parametri per accelerare la validazione --->
        # Esegui la validazione ogni N epoche. Impostare a 1 per validare dopo ogni epoca.
        "validate_every_n_epochs": 5,

        # Numero di esempi da usare per la validazione.
        # Utile per un feedback rapido durante lo sviluppo.
        # Impostare a -1 per usare l'INTERO set di validazione (per i risultati finali).
        "num_validation_examples": 100,
        # <--- FINE DELLE NUOVE AGGIUNTE --->

        "model_folder": "weights",
        "model_basename": "nanosocrates_",
        "preload": None,
        "tokenizer_file": "../tokenizer/nanosocrates_hf_tokenizer_1000.json",
        "corpus_file": "../dataset/training_corpus_1000.txt",
    }
