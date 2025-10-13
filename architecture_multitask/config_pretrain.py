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
        "num_validation_examples": -1,
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
        "model_folder": "weights_pretrain_nano_10k_data",
        "model_basename": "nanosocrates_nano_10k_data_pretrained_",
        "experiment_name": "runs/nanosocrates_nano_10k_data_pretrain",

        "preload": None,
        "scheduler_type": "linear_warmup",
        "warmup_percentage": 0.1,

        "loss_label_smoothing": 0.0,
    })
    return config


# --- NUOVA FASE 2: DECODER TUNING ---
def get_decoder_tuning_config():
    """
    FASE 2: Adattamento del solo Decoder (Encoder CONGELATO).
    Usa un LR medio per addestrare il decoder a interpretare le rappresentazioni dell'encoder.
    """
    config = get_nano_config()
    config['dropout'] = 0.15

    config.update({
        "num_epochs": 20,
        "lr": 5e-5,
        "validate_every_n_epochs": 4,
        "data_dir": "../dataset/training_data_cleaned",

        "model_folder": "weights_decoder_tuned_nano",
        "model_basename": "nanosocrates_decoder_tuned_nano_",
        "experiment_name": "runs/nanosocrates_decoder_tune_nano",

        "preload": "weights_decoder_tuned_nano/nanosocrates_decoder_tuned_nano_07.pt",

        "scheduler_type": "linear_warmup",
        "warmup_percentage": 0.1,

        "freeze_encoder": True,

        "loss_label_smoothing": 0.1,
    })
    return config


def get_full_finetune_config():
    """
    FASE 3: Fine-tuning completo End-to-End (Encoder SCONGELATO).
    Usa un LR molto basso per affinare l'intero modello senza oblio catastrofico.
    """
    config = get_nano_config()
    config['dropout'] = 0.1

    config.update({
        "num_epochs": 60,
        "lr": 1e-5,
        "validate_every_n_epochs": 3,
        "data_dir": "../dataset/training_data_cleaned",

        "model_folder": "weights_full_finetuned_nano",
        "model_basename": "nanosocrates_full_finetuned_nano_",
        "experiment_name": "runs/nanosocrates_full_finetune_nano",

        "preload": "weights_decoder_tuned_nano/nanosocrates_decoder_tuned_nano_19.pt",

        "scheduler_type": "linear_warmup",
        "warmup_percentage": 0.1,

        "freeze_encoder": False,

        "loss_label_smoothing": 0.1,
    })
    return config
