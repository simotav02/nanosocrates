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
        "num_validation_examples": -1,
        "tokenizer_file": "../tokenizer/film_corpus_bpe_final.json",
    }

def get_pretrain_config():
    """Configurazione per la fase di pre-training solo su MLM."""
    config = get_base_config()
    config.update({
        "num_epochs": 40,
        "lr": 5e-4,
        "validate_every_n_epochs": 5,
        "data_dir": "../dataset/pretrain_mlm_data",
        "model_folder": "weights_pretrain_mlm",
        "model_basename": "nanosocrates_mlm_pretrained_",
        "experiment_name": "runs/nanosocrates_pretrain_mlm",
        "preload": None
    })
    return config

def get_finetune_config():
    """Configurazione per la fase di fine-tuning su tutti i task."""
    config = get_base_config()
    config.update({
        "num_epochs": 30,
        "lr": 1e-4, # Learning rate più basso per il fine-tuning
        "validate_every_n_epochs": 2,
        "data_dir": "../dataset/finetune_all_tasks_data",
        "model_folder": "weights_finetuned",
        "model_basename": "nanosocrates_finetuned_",
        "experiment_name": "runs/nanosocrates_finetune_all_tasks",
        # --- PUNTO CHIAVE: Carica i pesi del miglior modello pre-addestrato ---
        "preload": "weights_pretrain_mlm/nanosocrates_mlm_pretrained_39.pt" # MODIFICA QUESTO NOME FILE!
    })
    return config