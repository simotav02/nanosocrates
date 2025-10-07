# config.py (precedentemente config_pretrain.py)

from pathlib import Path


def get_base_config():
    """
    Configurazione di base 'NANO' per combattere l'overfitting e accelerare il training.
    Condivisa tra pre-training e fine-tuning.
    """
    return {
        "batch_size": 16,  # Aumentato grazie al modello più piccolo
        "seq_len": 256,  # CRUCIALE: riduce drasticamente memoria e tempi
        "d_model": 256,  # Dimensione embedding/hidden state ridotta
        "N": 3,  # 3 layer encoder/decoder invece di 6
        "h": 4,  # Meno teste di attenzione
        "d_ff": 1024,  # Feed-forward dimension (solitamente 4 * d_model)
        "dropout": 0.1,
        "num_validation_examples": -1,  # Numero di esempi da usare per la validazione
        "tokenizer_file": "../tokenizer/film_corpus_bpe_tokenizer_t5.json",
    }


def get_pretrain_config():
    """Configurazione per la fase di PRE-TRAINING con Span Corruption."""
    config = get_base_config()
    config.update({
        "num_epochs": 50,
        "lr": 3e-4,  # Un learning rate più alto è corretto per il training da zero
        "validate_every_n_epochs": 10,
        "data_dir": "pretrain_t5_style_data",
        "model_folder": "weights_pretrain_t5",
        "model_basename": "nanosocrates_t5_pretrained_",
        "experiment_name": "runs/nanosocrates_pretrain_t5",
        "preload": None,  # Deve partire da zero
        "loss_label_smoothing": 0.0,  # NIENTE label smoothing per il pre-training
    })
    return config


def get_finetune_config():
    """Configurazione per la fase di FINE-TUNING sui 4 task specifici."""
    config = get_base_config()
    config.update({
        "num_epochs": 80,
        "lr": 2e-5,  # <-- LEARNING RATE CRUCIALE: più basso per un fine-tuning stabile
        "validate_every_n_epochs": 5,
        "data_dir": "../dataset/training_data_cleaned",
        "model_folder": "weights_finetuned_t5",
        "model_basename": "nanosocrates_t5_finetuned_",
        "experiment_name": "runs/nanosocrates_finetune_t5",

        # --- PUNTO CHIAVE: Carica i pesi del miglior modello PRE-ADDESTRATO ---
        # MODIFICA 'XX' con il numero di epoca del checkpoint migliore ottenuto dal pre-training!
        # Esempio: "weights_pretrain_t5/nanosocrates_t5_pretrained_49.pt"
        "preload": "weights_pretrain_t5/nanosocrates_t5_pretrained_49.pt",

        "loss_label_smoothing": 0.1,  # Usa il label smoothing per il fine-tuning per regolarizzare
    })
    return config