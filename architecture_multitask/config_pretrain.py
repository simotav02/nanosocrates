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
        # CORRETTO: Dalla cartella 'architecture_multitask', vai su di uno ('..') e poi dentro 'tokenizer'
        "tokenizer_file": "../tokenizer/film_corpus_bpe_tokenizer_t5.json",
    }

def get_pretrain_config():
    """Configurazione per la fase di PRE-TRAINING con Span Corruption."""
    config = get_base_config()
    config.update({
        "num_epochs": 50,
        "lr": 3e-4,
        "validate_every_n_epochs": 5,
        # MODIFICATO: Dalla cartella 'architecture_multitask', vai su di uno ('..') e poi dentro 'dataset_pretrain/pretrain_t5_style_data'
        "data_dir": "../dataset_pretrain/pretrain_t5_style_data",
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
        "num_epochs": 30,
        "lr": 4e-5,
        "validate_every_n_epochs": 5,
        # MODIFICATO: Dalla cartella 'architecture_multitask', vai su di uno ('..') e poi dentro 'dataset_pretrain/mlm_only_data'
        "data_dir": "../dataset_pretrain/mlm_only_data",
        "model_folder": "weights_task_adapt_t5",
        "model_basename": "nanosocrates_t5_task_adapt_",
        "experiment_name": "runs/nanosocrates_task_adapt_t5",
        # CORRETTO: Questo path è relativo alla cartella di esecuzione 'architecture_multitask'
        "preload": "weights_pretrain_t5/nanosocrates_t5_pretrained_09.pt",
        "loss_label_smoothing": 0.0,
    })
    return config

def get_finetune_config():
    """Configurazione per la fase di FINE-TUNING sui 4 task specifici."""
    config = get_base_config()
    config['dropout'] = 0.2
    config.update({
        "num_epochs": 60,
        "lr": 3e-5,
        "validate_every_n_epochs": 5,
        # CORRETTO: Dalla cartella 'architecture_multitask', vai su di uno ('..') e poi dentro 'dataset/training_data_cleaned'
        "data_dir": "../dataset/training_data_cleaned",
        "model_folder": "weights_finetuned_t5",
        "model_basename": "nanosocrates_t5_finetuned_",
        "experiment_name": "runs/nanosocrates_finetune_t5",
        # CORRETTO: Questo path è relativo alla cartella di esecuzione 'architecture_multitask'
        "preload": "weights_task_adapt_t5/nanosocrates_t5_task_adapt_29.pt",
        "loss_label_smoothing": 0.1,
    })
    return config