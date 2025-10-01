from pathlib import Path


def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 50,
        "lr": 5e-4,
        "seq_len": 512,
        "d_model": 256,
        "N": 4,  # Numero di blocchi Encoder/Decoder
        "h": 4,  # Numero di teste di attenzione
        "d_ff": 1024,  # Dimensione del layer FeedForward
        "dropout": 0.1,

        # --- Impostazioni per la validazione e il salvataggio ---
        "validate_every_n_epochs": 5,
        "num_validation_examples": -1,  # Esegui validazione su N esempi, -1 per l'intero set
        "model_folder": "weights_test_200_films",
        "model_basename": "nanosocrates_test_200",
        "preload": None,  # Imposta un percorso a un modello .pt per riprendere il training

        # --- Percorsi ai dati ---
        "data_dir": "../dataset/training_data_cleaned",
        "tokenizer_file": "../tokenizer/film_corpus_bpe_tokenizer.json",

        # --- Impostazioni per TensorBoard ---
        "experiment_name": "runs/nanosocrates_test_200_films_run"
    }