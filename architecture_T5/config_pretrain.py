from pathlib import Path

def get_base_config():
    """
    Configurazione 'NANO' per combattere l'overfitting.
     dropout aumentato e modello più piccolo.
    """
    return {
        "batch_size": 16,
        "seq_len": 256,
        "d_model": 256,
        "N": 2,                   # Ridotto a 2 layer per un modello ancora più semplice
        "h": 4,
        "d_ff": 1024,
        "dropout": 0.3,           # Aumentato in modo significativo per combattere l'overfitting
        "num_validation_examples": 500,
        "tokenizer_file": "../tokenizer/film_corpus_bpe_tokenizer_t5.json",
    }

def get_pretrain_config():
    """Configurazione per la fase di PRE-TRAINING con Span Corruption."""
    config = get_base_config()
    config.update({
        "num_epochs": 50,
        "lr": 3e-4,
        "validate_every_n_epochs": 5, # Validiamo più spesso per trovare il punto di early stopping
        "data_dir": "pretrain_t5_style_data",
        "model_folder": "weights_pretrain_t5",
        "model_basename": "nanosocrates_t5_pretrained_",
        "experiment_name": "runs/nanosocrates_pretrain_t5",
        "preload": None,
        "loss_label_smoothing": 0.0,
    })
    return config

def get_task_adapt_config():
    """Configurazione per la fase di ADATTAMENTO al task strutturato (MLM)."""
    config = get_base_config()
    config.update({
        "num_epochs": 30,         # Epoche sufficienti per imparare la sintassi
        "lr": 4e-5,               # Learning rate basso, stiamo solo adattando
        "validate_every_n_epochs": 5,
        "data_dir": "mlm_only_data", # <-- Usa i dati solo MLM
        "model_folder": "weights_task_adapt_t5",
        "model_basename": "nanosocrates_t5_task_adapt_",
        "experiment_name": "runs/nanosocrates_task_adapt_t5",
        # --- CARICA IL MIGLIOR MODELLO DAL PRE-TRAINING ---
        # MODIFICA 'XX' con il numero di epoca del checkpoint migliore!
        "preload": "weights_pretrain_t5/nanosocrates_t5_pretrained_XX.pt",
        "loss_label_smoothing": 0.0,
    })
    return config

def get_finetune_config():
    """Configurazione per la fase di FINE-TUNING sui 4 task specifici."""
    config = get_base_config()
    # Per il fine-tuning, un dropout leggermente più basso può aiutare
    config['dropout'] = 0.2
    config.update({
        "num_epochs": 60,         # Aumentato leggermente
        "lr": 2e-5,               # Learning rate molto basso
        "validate_every_n_epochs": 2,
        "data_dir": "../dataset/training_data_cleaned",
        "model_folder": "weights_finetuned_t5",
        "model_basename": "nanosocrates_t5_finetuned_",
        "experiment_name": "runs/nanosocrates_finetune_t5",
        # --- CARICA IL MIGLIOR MODELLO DAL TASK-ADAPTATION ---
        # MODIFICA 'YY' con il numero di epoca del checkpoint migliore!
        "preload": "weights_task_adapt_t5/nanosocrates_t5_task_adapt_YY.pt",
        "loss_label_smoothing": 0.1,
    })
    return config