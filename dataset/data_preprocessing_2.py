import json
import random
from tqdm import tqdm

# --- CONFIGURAZIONE ---
INPUT_JSON_FILE = "film_dataset_1000_final.json"
OUTPUT_DIR = "training_data"  # Creeremo file separati per input e output

# Token speciali (invariati)
SOT_TOKEN = "<SOT>"
EOT_TOKEN = "<EOT>"
SUBJ_TOKEN = "<SUBJ>"
PRED_TOKEN = "<PRED>"
OBJ_TOKEN = "<OBJ>"
MASK_TOKEN = "<MASK>"
TEXT_TO_RDF_TOKEN = "<Text2RDF>"
RDF_TO_TEXT_TOKEN = "<RDF2Text>"
CONTINUE_RDF_TOKEN = "<CONTINUERDF>"
MLM_TOKEN = "<MLM>"  # Aggiungiamo un token per il task MLM per coerenza


def linearize_triples(triples):
    """Converte una lista di triple in una singola stringa serializzata."""
    if not triples:
        return ""
    serialized_parts = []
    for triple in triples:
        part = (f"{SOT_TOKEN} {SUBJ_TOKEN} {triple['subject']} "
                f"{PRED_TOKEN} {triple['predicate']} "
                f"{OBJ_TOKEN} {triple['object']} {EOT_TOKEN}")
        serialized_parts.append(part)
    return " ".join(serialized_parts)


def main():
    """
    Funzione principale che legge il JSON, genera gli esempi bilanciati
    e li salva in file di training separati.
    """
    print(f"Caricamento dati da '{INPUT_JSON_FILE}'...")
    with open(INPUT_JSON_FILE, 'r', encoding='utf-8') as f:
        all_films_data = json.load(f)

    heavy_tasks_examples = []
    mlm_task_candidates = []

    print("Generazione degli esempi di addestramento (Passata 1: Task generativi)...")
    for film_data in tqdm(all_films_data, desc="Processando i film"):
        abstract = film_data.get("abstract", "").strip()
        triples = film_data.get("triples", [])

        if not abstract or not triples:
            continue

        # --- Task 1: Text2RDF ---
        input_text1 = f"{abstract} {TEXT_TO_RDF_TOKEN}"
        output_text1 = linearize_triples(triples)
        heavy_tasks_examples.append({"input": input_text1, "output": output_text1})

        # --- Task 2: RDF2Text ---
        input_text2 = f"{linearize_triples(triples)} {RDF_TO_TEXT_TOKEN}"
        output_text2 = abstract
        heavy_tasks_examples.append({"input": input_text2, "output": output_text2})

        # --- Task 4: RDF Completion 2 (CONTINUERDF) ---
        if len(triples) > 1:
            split_point = random.randint(1, len(triples) - 1)
            context_triples = triples[:split_point]
            completion_triples = triples[split_point:]

            input_text4 = f"{linearize_triples(context_triples)} {CONTINUE_RDF_TOKEN}"
            output_text4 = linearize_triples(completion_triples)
            heavy_tasks_examples.append({"input": input_text4, "output": output_text4})

        # --- Raccolta candidati per Task 3 (MLM) ---
        for triple in triples:
            # Mascheriamo l'oggetto (formato come da traccia, senza <OBJ>)
            mlm_task_candidates.append({
                "input": (f"{SOT_TOKEN} {SUBJ_TOKEN} {triple['subject']} "
                          f"{PRED_TOKEN} {triple['predicate']} {MASK_TOKEN} {EOT_TOKEN} {MLM_TOKEN}"),
                "output": triple['object']
            })
            # Mascheriamo il predicato (formato come da traccia, senza <PRED>)
            mlm_task_candidates.append({
                "input": (f"{SOT_TOKEN} {SUBJ_TOKEN} {triple['subject']} "
                          f"{MASK_TOKEN} {OBJ_TOKEN} {triple['object']} {EOT_TOKEN} {MLM_TOKEN}"),
                "output": triple['predicate']
            })

    print(f"Generati {len(heavy_tasks_examples)} esempi per i task generativi.")
    print(f"Generati {len(mlm_task_candidates)} candidati per il task MLM.")

    # --- Bilanciamento e Finalizzazione ---
    print("Bilanciamento del dataset...")
    # Mescoliamo i candidati MLM
    random.shuffle(mlm_task_candidates)

    # Selezioniamo un numero di esempi MLM pari al numero di altri task
    num_heavy_tasks = len(heavy_tasks_examples)
    selected_mlm_examples = mlm_task_candidates[:num_heavy_tasks]

    print(f"Selezionati {len(selected_mlm_examples)} esempi MLM per il bilanciamento.")

    # Uniamo e mescoliamo tutto
    final_training_data = heavy_tasks_examples + selected_mlm_examples
    random.shuffle(final_training_data)

    print(f"Dataset finale bilanciato con {len(final_training_data)} esempi totali.")

    # --- Salvataggio su file separati (formato comune per seq2seq) ---
    import os
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    source_filepath = os.path.join(OUTPUT_DIR, "train.source")
    target_filepath = os.path.join(OUTPUT_DIR, "train.target")

    print(f"Salvataggio del corpus in '{source_filepath}' e '{target_filepath}'...")
    with open(source_filepath, 'w', encoding='utf-8') as f_source, \
            open(target_filepath, 'w', encoding='utf-8') as f_target:
        for example in tqdm(final_training_data, desc="Scrivendo i file"):
            f_source.write(example["input"] + "\n")
            f_target.write(example["output"] + "\n")

    print("\nPROCESSO COMPLETATO.")
    print(f"File di training generati con successo nella cartella '{OUTPUT_DIR}'.")


if __name__ == "__main__":
    main()