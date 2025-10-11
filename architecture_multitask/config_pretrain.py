# config_pretrain.py (Versione a 2 Fasi con opzione "Micro")

from pathlib import Path


def get_nano_config():
    """
    Configurazione 'NANO' originale (20M params).
    Buona per un dataset di medie dimensioni.
    """
    return {
        "batch_size": 16,
        "seq_len": 256,
        "d_model": 256,
        "N": 2,
        "h": 4,
        "d_ff": 1024,  # 4 * d_model
        "dropout": 0.3,
        "num_validation_examples": 500,
        "tokenizer_file": "../tokenizer/film_corpus_bpe_tokenizer_t5.json",
    }


def get_micro_config():
    """
    Configurazione 'MICRO' ancora più piccola per combattere l'overfitting
    su dataset ridotti.
    d_model=128, d_ff=512.
    """
    return {
        "batch_size": 32,  # Aumentato perché il modello è più leggero e occupa meno memoria
        "seq_len": 256,
        "d_model": 128,  # Ridotto da 256
        "N": 2,  # Mantenuto a 2 (minimo ragionevole)
        "h": 4,  # Mantenuto a 4 (dim_head = 128/4 = 32, è un buon valore)
        "d_ff": 512,  # Ridotto da 1024 (4 * 128)
        "dropout": 0.3,  # Mantenuto alto per regolarizzazione
        "num_validation_examples": 500,
        "tokenizer_file": "../tokenizer/film_corpus_bpe_tokenizer_t5.json",
    }


# in config_pretrain.py

def get_pretrain_config():
    """Configurazione per la fase di PRE-TRAINING con Span Corruption."""

    config = get_micro_config()

    config.update({
        "num_epochs": 40,  # Estendiamo il training di 15 epoche partendo da ~25
        "lr": 5e-5,  # Riduciamo il learning rate come pianificato
        "validate_every_n_epochs": 2,
        "data_dir": "../dataset_pretrain/pretrain_t5_style_data",
        "model_folder": "weights_pretrain_micro_2_linear",  # Nuova cartella per non confondere i modelli
        "model_basename": "nanosocrates_micro_pretrained_linear_",
        "experiment_name": "runs/nanosocrates_micro_pretrain_2_linear",

        # --- MODIFICHE CHIAVE ---
        "preload": "weights_pretrain_micro_2/nanosocrates_micro_pretrained_23.pt",  # Carica il tuo checkpoint migliore
        "scheduler_type": "linear_warmup",  # Specifica il nuovo scheduler
        "warmup_percentage": 0.1,  # Percentuale di step totali per il warmup (standard)
        # --- FINE MODIFICHE ---

        "loss_label_smoothing": 0.0,
    })
    return config


# in config_pretrain.py

def get_finetune_config():
    """Configurazione per la fase di FINE-TUNING sui 4 task specifici."""

    # ... (scegli la config nano o micro)
    config = get_micro_config()
    config['dropout'] = 0.25

    config.update({
        "num_epochs": 25,
        "lr": 2e-5,
        "validate_every_n_epochs": 5,
        "data_dir": "../dataset/training_data_cleaned",
        "model_folder": "weights_finetuned_micro",
        "model_basename": "nanosocrates_micro_finetuned_",
        "experiment_name": "runs/nanosocrates_micro_finetune",

        # Aggiorna il preload con il percorso della cartella del pre-training che deciderai di usare alla fine
        "preload": "weights_pretrain_micro_2/nanosocrates_micro_pretrained_27.pt",

        # --- MODIFICA PER COERENZA ---
        "scheduler_type": "linear_warmup",
        "warmup_percentage": 0.1,
        # --- FINE MODIFICA ---

        "loss_label_smoothing": 0.1,
    })
    return config