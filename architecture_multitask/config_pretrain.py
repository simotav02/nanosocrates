from pathlib import Path


def get_nano_config():
    """
    Configurazione 'NANO'
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
        "tokenizer_file": "../tokenizer/film_corpus_bpe_tokenizer_t5_2.json",
    }


def get_micro_config():
    """
    Configurazione 'MICRO'
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
        "tokenizer_file": "../tokenizer/film_corpus_bpe_tokenizer_t5_2.json",
    }


def get_pretrain_config():
    """Configurazione per la fase di PRE-TRAINING con Span Corruption."""

    config = get_nano_config()

    config.update({
        "num_epochs": 30,
        "lr": 1e-4,
        "validate_every_n_epochs": 2,
        "data_dir": "../dataset_pretrain/pretrain_t5_style_data_v2",
        "model_folder": "weights_prova_3/weights_pretrain_prova_3",
        "model_basename": "nanosocrates_prova_3_pretrained_",
        "experiment_name": "runs/nanosocrates_pretrain_prova_3",

        "preload": None,
        "scheduler_type": "linear_warmup",
        "warmup_percentage": 0.1,

        "loss_label_smoothing": 0.0,
    })
    return config


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
        "data_dir": "../dataset/training_data_cleaned_2",

        "model_folder": "weights_prova_3/weights_decoder_tuned_prova_3",
        "model_basename": "nanosocrates_prova_3_decoder_tuned_",
        "experiment_name": "runs/nanosocrates_decoder_tune_prova_3",

        "preload": "weights_prova_3/weights_pretrain_prova_3/nanosocrates_prova_3_pretrained_07.pt",

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
        "data_dir": "../dataset/training_data_cleaned_2",

        "model_folder": "weights_prova_3/weights_full_finetuned_prova_3",
        "model_basename": "nanosocrates_prova_3_full_finetuned_",
        "experiment_name": "runs/nanosocrates_full_finetune_prova_3",

        "preload": "weights_decoder_tuned_prova_1/nanosocrates_decoder_tuned_nano_XX.pt",

        "scheduler_type": "linear_warmup",
        "warmup_percentage": 0.1,

        "freeze_encoder": False,

        "loss_label_smoothing": 0.1,
    })
    return config
