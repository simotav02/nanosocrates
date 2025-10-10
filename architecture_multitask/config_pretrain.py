# config_pretrain.py (Versione a 2 Fasi)

from pathlib import Path

def get_base_config():
    """
    Configurazione 'NANO' per combattere l'overfitting.
    """
    return {
        "batch_size": 16,
        "seq_len": 256,
        "d_model": 256,
        "N": 2,
        "h": 4,
        "d_ff": 1024,
        "dropout": 0.3,
        "num_validation_examples": 500,
        "tokenizer_file": "../tokenizer/film_corpus_bpe_tokenizer_t5.json",
    }

def get_pretrain_config():
    """Configurazione per la fase di PRE-TRAINING con Span Corruption."""
    config = get_base_config()
    # Per il pre-training, un LR più alto e validazione meno frequente vanno bene.
    # Monitora la validation loss per trovare il punto di early stopping.
    config.update({
        "num_epochs": 50,
        "lr": 1e-4, # Un LR più conservativo che ha mostrato di funzionare meglio
        "validate_every_n_epochs": 2, # Validiamo più spesso per trovare il punto ottimale
        "data_dir": "../dataset_pretrain/pretrain_t5_style_data",
        "model_folder": "weights_pretrain_unified",
        "model_basename": "nanosocrates_unified_pretrained_",
        "experiment_name": "runs/nanosocrates_unified_pretrain",
        "preload": None,
        "loss_label_smoothing": 0.0,
    })
    return config

def get_finetune_config():
    """Configurazione per la fase di FINE-TUNING sui 4 task specifici."""
    config = get_base_config()
    # Per il fine-tuning, un dropout leggermente più basso e un LR molto basso sono cruciali.
    config['dropout'] = 0.25
    config.update({
        "num_epochs": 25, # Partiamo con meno epoche per evitare overfitting rapido
        "lr": 3e-5,
        "validate_every_n_epochs": 1, # Validiamo ad OGNI epoca
        "data_dir": "../dataset/training_data_cleaned",
        "model_folder": "weights_finetuned_unified",
        "model_basename": "nanosocrates_unified_finetuned_",
        "experiment_name": "runs/nanosocrates_unified_finetune",
        # --- CARICA IL MIGLIOR MODELLO DIRETTAMENTE DAL PRE-TRAINING ---
        # MODIFICA 'XX' con il numero di epoca del checkpoint migliore!
        "preload": "weights_pretrain_unified/nanosocrates_unified_pretrained_XX.pt", # Esempio: _09.pt
        "loss_label_smoothing": 0.1,
    })
    return config