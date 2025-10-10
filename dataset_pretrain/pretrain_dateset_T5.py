import os
import random
from tokenizers import Tokenizer
from tqdm import tqdm

try:
    import numpy as np
except ImportError:
    raise ImportError("Questo script richiede NumPy. Per favore, installalo con: pip install numpy")

INPUT_CORPUS_FILE = "./pretrain_corpus_data/pretrain_corpus.txt"
TOKENIZER_PATH = "../tokenizer/film_corpus_bpe_tokenizer_t5.json"
OUTPUT_DIR = "pretrain_t5_style_data"

CORRUPTION_RATE = 0.15


def t5_span_corruption(text: str, tokenizer: Tokenizer, noise_density: float):
    """
    Versione 3: Approccio robusto che usa segnaposto per la corruzione.
    1.  Tokenizza il testo in ID.
    2.  Seleziona casualmente gli ID da mascherare.
    3.  Raggruppa gli indici consecutivi in span.
    4.  Crea una copia della lista di ID.
    5.  Itera sugli span:
        a. Costruisci il target aggiungendo <extra_id_X> e i token originali dello span.
        b. Modifica la copia della lista di input: sostituisci il primo token dello span
           con <extra_id_X> e tutti gli altri con un segnaposto (-1).
    6.  Filtra la lista di input per rimuovere tutti i segnaposto (-1).
    7.  Decodifica le due liste di ID finali.
    """
    encoding = tokenizer.encode(text)
    original_ids = encoding.ids
    n_tokens = len(original_ids)

    if n_tokens < 2:
        return text, ""

    num_noise_tokens = round(n_tokens * noise_density)
    num_noise_tokens = min(max(num_noise_tokens, 1), n_tokens - 1)

    indices_to_mask = sorted(list(np.random.permutation(n_tokens)[:num_noise_tokens]))

    if not indices_to_mask:
        return text, ""

    # Raggruppa gli indici consecutivi in span
    spans_indices = []
    if indices_to_mask:
        current_span = [indices_to_mask[0]]
        for i in range(1, len(indices_to_mask)):
            if indices_to_mask[i] == indices_to_mask[i - 1] + 1:
                current_span.append(indices_to_mask[i])
            else:
                spans_indices.append(current_span)
                current_span = [indices_to_mask[i]]
        spans_indices.append(current_span)

    # Inizializza le liste di ID
    input_ids_list = list(original_ids)
    target_ids_list = []
    extra_id_counter = 0

    for span in spans_indices:
        if extra_id_counter >= 100: break

        extra_id_token = f"<extra_id_{extra_id_counter}>"
        extra_id_token_id = tokenizer.token_to_id(extra_id_token)
        if extra_id_token_id is None:
            print(f"ATTENZIONE: Token sentinella non trovato: {extra_id_token}")
            continue

        # Aggiungi lo span al target
        target_ids_list.append(extra_id_token_id)
        target_ids_list.extend([original_ids[i] for i in span])

        # Sostituisci il primo token dello span nell'input con la sentinella
        input_ids_list[span[0]] = extra_id_token_id
        # Marca gli altri token dello span per la rimozione
        for i in span[1:]:
            input_ids_list[i] = -1

        extra_id_counter += 1

    if not target_ids_list:
        return text, ""

    # Filtra gli ID marcati per la rimozione (-1)
    final_input_ids = [token_id for token_id in input_ids_list if token_id != -1]

    # Decodifica le liste di ID una sola volta alla fine.
    # `skip_special_tokens=False` è essenziale.
    corrupted_text = tokenizer.decode(final_input_ids, skip_special_tokens=False)
    target_text = tokenizer.decode(target_ids_list, skip_special_tokens=False)

    return corrupted_text, target_text


def main():
    print(f"--- Creazione del Dataset di Pre-training stile T5 (Approccio Puro) ---")

    if not os.path.exists(TOKENIZER_PATH):
        print(f"ERRORE: Tokenizer non trovato in '{TOKENIZER_PATH}'.")
        return
    print(f"1/4 - Caricamento del tokenizer da '{TOKENIZER_PATH}'...")
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

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
            line, tokenizer, CORRUPTION_RATE
        )
        if corrupted_input and target:
            corrupted_examples.append({"input": corrupted_input, "output": target})

    print(f"4/4 - Salvataggio del nuovo dataset in '{OUTPUT_DIR}'...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

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