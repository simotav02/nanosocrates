# config.py

from pathlib import Path

def get_base_config():
    """Configurazione di base condivisa tra pre-training e fine-tuning."""
    return {
        "batch_size": 8,
        "seq_len": 512,
        "d_model": 512,
        "N": 6,
        "h": 8,
        "d_ff": 2048,
        "dropout": 0.1,
        "num_validation_examples": 500, # Riduci per validazioni più veloci
        "tokenizer_file": "../tokenizer/film_corpus_bpe_tokenizer_t5.json",
    }

def get_pretrain_config():
    """Configurazione per la fase di PRE-TRAINING con Span Corruption."""
    config = get_base_config()
    config.update({
        "num_epochs": 50,         # Un numero robusto di epoche per il pre-training
        "lr": 3e-4,               # Un learning rate più alto per il training da zero
        "validate_every_n_epochs": 10,
        "data_dir": "pretrain_t5_style_data", # <-- Usa i dati corrotti
        "model_folder": "weights_pretrain_t5",
        "model_basename": "nanosocrates_t5_pretrained_",
        "experiment_name": "runs/nanosocrates_pretrain_t5",
        "preload": None,          # Deve partire da zero
        "loss_label_smoothing": 0.0, # NIENTE label smoothing per il pre-training
    })
    return config

def get_finetune_config():
    """Configurazione per la fase di FINE-TUNING sui 4 task specifici."""
    config = get_base_config()
    config.update({
        "num_epochs": 30,         # Meno epoche per la fase di adattamento
        "lr": 3e-5,               # <-- LEARNING RATE CRUCIALE: molto più basso!
        "validate_every_n_epochs": 5,
        "data_dir": "dataset/unified_multitask_data", # <-- Usa i dati dei task finali
        "model_folder": "weights_finetuned_t5",
        "model_basename": "nanosocrates_t5_finetuned_",
        "experiment_name": "runs/nanosocrates_finetune_t5",
        # --- PUNTO CHIAVE: Carica i pesi del miglior modello pre-addestrato ---
        # MODIFICA QUESTO NOME FILE con il checkpoint migliore ottenuto dal pre-training!
        "preload": "weights_pretrain_t5/nanosocrates_t5_pretrained_49.pt",
        "loss_label_smoothing": 0.1, # Usa il label smoothing per il fine-tuning
    })
    return config