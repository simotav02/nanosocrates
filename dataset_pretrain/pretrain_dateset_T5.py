# pretrain_dateset_T5.py (Versione Modificata)

import os
import random
from tokenizers import Tokenizer
from tqdm import tqdm

try:
    import numpy as np
except ImportError:
    raise ImportError("Questo script richiede NumPy. Per favore, installalo con: pip install numpy")

# --- MODIFICA 1: CAMBIA L'INPUT ---
# INPUT_DATA_DIR = "../dataset/training_data_cleaned" # Vecchio input
INPUT_CORPUS_FILE = "./pretrain_corpus_data/pretrain_corpus.txt"  # Nuovo input

TOKENIZER_PATH = "../tokenizer/film_corpus_bpe_tokenizer_t5.json"
OUTPUT_DIR = "pretrain_t5_style_data"

CORRUPTION_RATE = 0.15
MEAN_NOISE_SPAN_LENGTH = 3.0


def t5_span_corruption(text: str, tokenizer: Tokenizer, noise_density: float, mean_noise_span_length: float):
    """
    Versione corretta della corruzione a span, che gestisce correttamente la
    decodifica dei token per evitare la corruzione dei dati.
    """
    original_tokens = tokenizer.encode(text).tokens
    n_tokens = len(original_tokens)

    if n_tokens < 2:
        return text, ""

    num_noise_tokens = round(n_tokens * noise_density)
    num_noise_tokens = min(max(num_noise_tokens, 1), n_tokens - 1)

    # Scegli gli indici dei token da mascherare
    indices_to_mask = sorted(np.random.permutation(n_tokens)[:num_noise_tokens])

    if not indices_to_mask:
        return text, ""

    # Raggruppa gli indici consecutivi in "span"
    spans = []
    current_span = [indices_to_mask[0]]
    for i in range(1, len(indices_to_mask)):
        if indices_to_mask[i] == indices_to_mask[i - 1] + 1:
            current_span.append(indices_to_mask[i])
        else:
            spans.append(current_span)
            current_span = [indices_to_mask[i]]
    spans.append(current_span)

    # Unisci span piccoli per raggiungere una lunghezza media desiderata
    num_spans_to_keep = max(1, round(num_noise_tokens / mean_noise_span_length))

    while len(spans) > num_spans_to_keep:
        # Trova la distanza più piccola tra due span e li unisce
        min_dist = float('inf')
        merge_idx = -1
        for i in range(len(spans) - 1):
            dist = spans[i + 1][0] - spans[i][-1]
            if dist < min_dist:
                min_dist = dist
                merge_idx = i

        # Unisci gli span e tutto ciò che c'è in mezzo
        end_of_first_span = spans[merge_idx][-1]
        start_of_second_span = spans[merge_idx + 1][0]
        merged_span = spans[merge_idx] + list(range(end_of_first_span + 1, start_of_second_span)) + spans[merge_idx + 1]

        spans[merge_idx] = merged_span
        del spans[merge_idx + 1]

    input_parts = []
    target_parts = []
    extra_id_counter = 0
    last_processed_idx = -1

    for span_indices in spans:
        if extra_id_counter >= 150: break  # Limite di sicurezza

        start_span_idx = span_indices[0]
        end_span_idx = span_indices[-1]

        # Aggiungi la parte di testo non mascherata prima dello span corrente
        input_parts.append("".join(original_tokens[last_processed_idx + 1: start_span_idx]))

        # Aggiungi il token di maschera all'input
        input_parts.append(f"<extra_id_{extra_id_counter}>")

        # Aggiungi il token di maschera e il testo mascherato al target
        target_parts.append(f"<extra_id_{extra_id_counter}>")
        target_parts.append("".join(original_tokens[start_span_idx: end_span_idx + 1]))

        last_processed_idx = end_span_idx
        extra_id_counter += 1

    # Aggiungi l'ultima parte di testo non mascherata
    input_parts.append("".join(original_tokens[last_processed_idx + 1:]))

    # Aggiungi l'ultimo token di maschera al target
    if extra_id_counter < 150:
        target_parts.append(f"<extra_id_{extra_id_counter}>")

    # Ricostruisci le stringhe finali
    corrupted_text = "".join(input_parts).replace(' ', ' ').strip()
    target_text = "".join(target_parts).replace(' ', ' ').strip()

    # Pulizia per il BPE: il tokenizer ByteLevel non ama gli spazi all'inizio
    # Questa pulizia è specifica per come funziona il tokenizer huggingface
    corrupted_text = corrupted_text.replace(' ', ' ').strip()
    target_text = target_text.replace(' ', ' ').strip()

    return corrupted_text, target_text


def main():
    print(f"--- Creazione del Dataset di Pre-training stile T5 (Approccio Puro) ---")

    if not os.path.exists(TOKENIZER_PATH):
        print(f"ERRORE: Tokenizer non trovato in '{TOKENIZER_PATH}'.")
        return
    print(f"1/4 - Caricamento del tokenizer da '{TOKENIZER_PATH}'...")
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

    # --- MODIFICA 2: CAMBIA LA LOGICA DI LETTURA ---
    print(f"2/4 - Lettura del corpus di origine puro da '{INPUT_CORPUS_FILE}'...")
    if not os.path.exists(INPUT_CORPUS_FILE):
        print(f"ERRORE: File del corpus puro non trovato. Esegui prima 'create_pretrain_corpus.py'.")
        return

    with open(INPUT_CORPUS_FILE, 'r', encoding='utf-8') as f:
        corpus_lines = f.readlines()

    corpus_lines = [line.strip() for line in corpus_lines if line.strip()]
    print(f"Trovate {len(corpus_lines)} righe totali nel corpus puro.")

    print("3/4 - Applicazione dello Span Corruption a ogni riga del corpus...")
    corrupted_examples = []
    for line in tqdm(corpus_lines, desc="Corrompendo il corpus"):
        corrupted_input, target = t5_span_corruption(
            line, tokenizer, CORRUPTION_RATE, MEAN_NOISE_SPAN_LENGTH
        )
        if corrupted_input and target:
            corrupted_examples.append({"input": corrupted_input, "output": target})

    # ... (Il resto dello script, da "4/4 - Salvataggio...", rimane INVARIATO) ...
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