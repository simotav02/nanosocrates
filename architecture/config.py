from pathlib import Path


def get_config():
    """
    Restituisce la configurazione per il training del modello NanoSocrates.
    I percorsi sono relativi alla directory principale del progetto.
    """
    return {
        "batch_size": 32,
        "num_epochs": 50,
        "lr": 1e-5,
        "seq_len": 512,
        "d_model": 512,
        "N": 4,  # Numero di blocchi Encoder/Decoder
        "h": 8,  # Numero di teste di attenzione
        "d_ff": 2048,  # Dimensione del layer FeedForward
        "dropout": 0.1,

        # --- Impostazioni per la validazione e il salvataggio ---
        "validate_every_n_epochs": 1,
        "num_validation_examples": 200,  # Esegui validazione su N esempi, -1 per l'intero set
        "model_folder": "weights",
        "model_basename": "nanosocrates_",
        "preload": None,  # Imposta un percorso a un modello .pt per riprendere il training

        # --- Percorsi ai dati ---
        # NOTA: Sostituito 'corpus_file' con 'data_dir' per adattarsi al nuovo Dataset
        "data_dir": "dataset/training_data",
        "tokenizer_file": "tokenizer/film_corpus_bpe_tokenizer.json",

        # --- Impostazioni per TensorBoard ---
        "experiment_name": "runs/nanosocrates_v2"
    }