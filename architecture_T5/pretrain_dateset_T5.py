# create_pretrain_dataset_t5.py

import os
import random
from tokenizers import Tokenizer
from tqdm import tqdm

try:
    import numpy as np
except ImportError:
    raise ImportError("Questo script richiede NumPy. Per favore, installalo con: pip install numpy")

# --- CONFIGURAZIONE ---

# 1. Percorso del dataset BILANCIATO che contiene TUTTI i task.
#    Useremo sia gli input che gli output di questo dataset come materiale
#    per il nostro pre-training (testo libero + triple linearizzate).
INPUT_DATA_DIR = "../dataset/training_data_cleaned"  # O unified_multitask_data

# 2. Percorso del tokenizer addestrato con i token <extra_id_...>.
TOKENIZER_PATH = "../tokenizer/film_corpus_bpe_tokenizer_t5.json"

# 3. Cartella dove verranno salvati i nuovi file di pre-training.
OUTPUT_DIR = "pretrain_t5_style_data"

# 4. Parametri per lo Span Corruption (valori standard dal paper di T5).
CORRUPTION_RATE = 0.15
MEAN_NOISE_SPAN_LENGTH = 3.0


# SOSTITUISCI la vecchia funzione con questa
def t5_span_corruption(text: str, tokenizer: Tokenizer, noise_density: float, mean_noise_span_length: float):
    original_tokens = tokenizer.encode(text).ids
    n_tokens = len(original_tokens)

    if n_tokens < 2:
        return text, ""

    num_noise_tokens = round(n_tokens * noise_density)
    num_noise_tokens = min(max(num_noise_tokens, 1), n_tokens - 1)

    # Usiamo numpy per un campionamento più efficiente e una generazione di lunghezze di span più accurata
    indices_to_mask = np.random.permutation(n_tokens)[:num_noise_tokens]

    # Calcola il numero di span rumorosi da generare
    num_spans = round(num_noise_tokens / mean_noise_span_length)
    num_spans = max(num_spans, 1)

    # Dividi gli indici in 'num_spans' gruppi
    split_indices = np.array_split(np.sort(indices_to_mask), num_spans)

    input_ids = list(original_tokens)
    target_spans = []
    extra_id_counter = 0

    # Raggruppa gli indici in span consecutivi
    for span_indices in split_indices:
        if len(span_indices) == 0:
            continue

        # Inserisci il token extra_id nel target
        extra_id_token_str = f"<extra_id_{extra_id_counter}>"
        extra_id_token_id = tokenizer.token_to_id(extra_id_token_str)

        # CONTROLLO DI SICUREZZA
        if extra_id_token_id is None:
            raise ValueError(
                f"CRITICO: Impossibile trovare l'ID per il token '{extra_id_token_str}'. Il tokenizer è corrotto.")

        target_spans.extend([extra_id_token_id] + [original_tokens[i] for i in span_indices])

        # Per l'input, dobbiamo sostituire il primo token dello span con l'extra_id
        # e marcare gli altri per la rimozione
        start_index = span_indices[0]
        input_ids[start_index] = extra_id_token_id
        for i in span_indices[1:]:
            input_ids[i] = -1  # Marcatore per la rimozione

        extra_id_counter += 1

    # Rimuovi i token marcati da input_ids
    filtered_input_ids = [token_id for token_id in input_ids if token_id != -1]

    # Finalizza il target
    final_extra_id_token_id = tokenizer.token_to_id(f"<extra_id_{extra_id_counter}>")
    target_spans.append(final_extra_id_token_id)

    corrupted_text = tokenizer.decode(filtered_input_ids, skip_special_tokens=False)
    target_text = tokenizer.decode(target_spans, skip_special_tokens=False)

    return corrupted_text, target_text


def main():
    """
    Funzione principale che orchestra la creazione del dataset di pre-training.
    """
    print(f"--- Creazione del Dataset di Pre-training stile T5 ---")

    # 1. Carica il tokenizer
    if not os.path.exists(TOKENIZER_PATH):
        print(f"ERRORE: Tokenizer non trovato in '{TOKENIZER_PATH}'. Esegui prima lo script del tokenizer.")
        return
    print(f"1/4 - Caricamento del tokenizer da '{TOKENIZER_PATH}'...")
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

    # 2. Raccogli tutto il testo disponibile (sia source che target dal dataset misto)
    print(f"2/4 - Lettura del corpus di origine da '{INPUT_DATA_DIR}'...")
    source_file = os.path.join(INPUT_DATA_DIR, "train.source")
    target_file = os.path.join(INPUT_DATA_DIR, "train.target")

    if not os.path.exists(source_file) or not os.path.exists(target_file):
        print(f"ERRORE: File del corpus non trovati in '{INPUT_DATA_DIR}'. Esegui prima lo script di preprocessing.")
        return

    corpus_lines = []
    with open(source_file, 'r', encoding='utf-8') as f:
        corpus_lines.extend(f.readlines())
    with open(target_file, 'r', encoding='utf-8') as f:
        corpus_lines.extend(f.readlines())

    # Rimuovi eventuali righe vuote e spaziature extra
    corpus_lines = [line.strip() for line in corpus_lines if line.strip()]
    print(f"Trovate {len(corpus_lines)} righe totali nel corpus.")

    # 3. Applica la corruzione e raccogli gli esempi
    print("3/4 - Applicazione dello Span Corruption a ogni riga del corpus...")
    corrupted_examples = []
    for line in tqdm(corpus_lines, desc="Corrompendo il corpus"):
        corrupted_input, target = t5_span_corruption(
            line, tokenizer, CORRUPTION_RATE, MEAN_NOISE_SPAN_LENGTH
        )
        if corrupted_input and target:
            corrupted_examples.append({"input": corrupted_input, "output": target})

    # 4. Salva il nuovo dataset
    print(f"4/4 - Salvataggio del nuovo dataset in '{OUTPUT_DIR}'...")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    source_out_path = os.path.join(OUTPUT_DIR, "train.source")
    target_out_path = os.path.join(OUTPUT_DIR, "train.target")

    random.shuffle(corrupted_examples)

    with open(source_out_path, 'w', encoding='utf-8') as f_source, \
            open(target_out_path, 'w', encoding='utf-8') as f_target:
        for example in tqdm(corrupted_examples, desc="Scrivendo i file"):
            f_source.write(example['input'] + '\n')
            f_target.write(example['output'] + '\n')

    print("\n--- PROCESSO COMPLETATO ---")
    print(f"Creati {len(corrupted_examples)} esempi di pre-training.")
    print(f"File salvati in '{source_out_path}' e '{target_out_path}'.")
    print("Ora puoi usare questa cartella per la fase di pre-training del tuo modello.")


if __name__ == "__main__":
    main()