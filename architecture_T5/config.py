def get_config():
    """
    Configurazione per un singolo ciclo di training multi-task unificato.
    Questo approccio è più semplice e robusto del pre-training/fine-tuning separato.
    """
    return {
        # --- Parametri del Modello ---
        "d_model": 512,
        "N": 6,  # Numero di blocchi Encoder/Decoder (standard)
        "h": 8,  # Numero di teste di attenzione
        "d_ff": 2048,  # Dimensione del layer FeedForward
        "dropout": 0.1,
        "seq_len": 512,

        # --- Parametri di Training ---
        "batch_size": 8,
        "num_epochs": 50,  # Un numero ragionevole di epoche per un dataset più grande
        "lr": 1e-4,  # Un learning rate più stabile per un training from-scratch

        # --- Percorsi e Nomi File ---
        # NOTA: Assicurati che questa cartella contenga i file train.source/target
        # generati dal nuovo dataset pulito e più grande.
        "data_dir": "../dataset/training_data_cleaned",
        "tokenizer_file": "../tokenizer/film_corpus_bpe_tokenizer.json",
        "model_folder": "weights_test_2000_films",
        "model_basename": "nanosocrates_test_2000_",

        # --- Checkpoint e Ripresa del Training ---
        # Imposta un percorso a un modello .pt per riprendere il training (es. "weights_unified_training/nanosocrates_unified_09.pt")
        "preload": None,

        # --- Validazione e Logging ---
        "validate_every_n_epochs": 5,
        "num_validation_examples": -1,  # -1 per validare sull'intero validation set
        "experiment_name": "runs/nanosocrates_test_2000_fims_run"
    }

































# from pathlib import Path
#
# def get_base_config():
#     """Configurazione di base condivisa tra pre-training e fine-tuning."""
#     return {
#         "batch_size": 8,
#         "seq_len": 512,
#         "d_model": 512,
#         "N": 4,
#         "h": 8,
#         "d_ff": 2048,
#         "dropout": 0.1,
#         "num_validation_examples": -1,
#         "tokenizer_file": "../tokenizer/film_corpus_bpe_tokenizer.json",
#     }
#
# def get_pretrain_config():
#     """Configurazione per la fase di pre-training solo su MLM."""
#     config = get_base_config()
#     config.update({
#         "num_epochs": 60,
#         "lr": 4e-4,
#         "validate_every_n_epochs": 10,
#         "data_dir": "../dataset/pretrain_mlm_data",
#         "model_folder": "weights_pretrain_mlm",
#         "model_basename": "nanosocrates_mlm_pretrained_",
#         "experiment_name": "runs/nanosocrates_pretrain_mlm",
#         "preload": None
#     })
#     return config
#
# # In config.py, modifica la funzione get_finetune_config()
#
# # def get_finetune_config():
# #     config = get_base_config()
# #     config.update({
# #         "num_epochs": 30,         # Epoche per la fase di adattamento
# #         "lr": 2e-5,               # <-- LEARNING RATE CRUCIALE: più basso!
# #         "validate_every_n_epochs": 5,
# #         "data_dir": "../dataset/finetune_all_tasks_data",
# #         "model_folder": "weights_mixtuning", # Nuova cartella
# #         "model_basename": "nanosocrates_mixtuning_",
# #         "experiment_name": "runs/nanosocrates_mixtuning",
# #         "preload": "weights_pretrain_mlm/nanosocrates_mlm_pretrained_59.pt" # Carica dal pre-training
# #     })
# #     return config
#
# def get_finetune_config():
#     """Configurazione per la fase di fine-tuning su tutti i task."""
#     config = get_base_config()
#     config.update({
#         "num_epochs": 80,
#         "lr": 2e-5, # Learning rate più basso per il fine-tuning
#         "validate_every_n_epochs": 5,
#         "data_dir": "../dataset/finetune_all_tasks_data",
#         "model_folder": "weights_finetuned",
#         "model_basename": "nanosocrates_finetuned_",
#         "experiment_name": "runs/nanosocrates_finetune_all_tasks",
#         # --- PUNTO CHIAVE: Carica i pesi del miglior modello pre-addestrato ---
#         "preload": "weights_pretrain_mlm/nanosocrates_mlm_pretrained_59.pt" # MODIFICA QUESTO NOME FILE!
#     })
#     return config
