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
        "d_ff": 1024,
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
        "batch_size": 32,
        "seq_len": 256,
        "d_model": 128,
        "N": 2,
        "h": 4,
        "d_ff": 512,
        "dropout": 0.3,
        "num_validation_examples": -1,
        "tokenizer_file": "../tokenizer/film_corpus_bpe_tokenizer_t5.json",
    }


def get_pretrain_config():
    """Configurazione per la fase di PRE-TRAINING con Span Corruption."""

    config = get_nano_config()

    config.update({
        "num_epochs": 60,
        "lr": 1e-4,
        "validate_every_n_epochs": 2,
        "data_dir": "../dataset_pretrain/pretrain_t5_style_data_v3",
        "model_folder": "weights_pretrain_nano_10_data",
        "model_basename": "nanosocrates_nano_10_data_pretrained_",
        "experiment_name": "runs/nanosocrates_nano_10_data_pretrain",

        "preload": None,
        "scheduler_type": "linear_warmup",
        "warmup_percentage": 0.1,

        "loss_label_smoothing": 0.0,
    })
    return config


def get_finetune_config():
    """Configurazione per la fase di FINE-TUNING sui 4 task specifici."""

    config = get_micro_config()
    config['dropout'] = 0.15

    config.update({
        "num_epochs": 25,
        "lr": 3e-5,
        "validate_every_n_epochs": 5,
        "data_dir": "../dataset/training_data_cleaned",
        "model_folder": "weights_finetuned_micro",
        "model_basename": "nanosocrates_micro_finetuned_",
        "experiment_name": "runs/nanosocrates_micro_finetune",
        "preload": "weights_pretrain_micro_hq_data/nanosocrates_micro_hq_data_pretrained_45.pt",

        "scheduler_type": "cosine_restarts",
        "warmup_percentage": 0.1,

        "loss_label_smoothing": 0.1,
    })
    return config