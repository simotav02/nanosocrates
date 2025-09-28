# data_preprocessing_2.py (MODIFICATO PER UN CORRETTO ALLINEAMENTO DEI DATI)

import json
import random
from tqdm import tqdm
import os
import re

# --- CONFIGURAZIONE ---
INPUT_JSON_FILE = "film_dataset_1000_final.json"
OUTPUT_DIR = "training_data"

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
MLM_TOKEN = "<MLM>"


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


# --- NUOVA FUNZIONE DI CONTROLLO PER L'ALLINEAMENTO ---
def is_entity_in_text(entity: str, text: str) -> bool:
    if not isinstance(entity, str):
        return False

    # 1. Rimuove il prefisso (es. 'dbr:')
    entity_name = entity.split(':')[-1]
    # 2. Sostituisce gli underscore con spazi
    entity_name = entity_name.replace('_', ' ')
    # 3. Rimuove eventuali qualificatori tra parentesi (es. "(film)")
    entity_name = re.sub(r'\(.*?\)', '', entity_name).strip()

    # Se il nome dell'entità è vuoto dopo la pulizia, non possiamo validarlo.
    if not entity_name:
        return False

    # 4. Controlla se tutte le parole del nome dell'entità sono presenti nel testo (case-insensitive)
    #    Questo è un controllo più robusto di una semplice ricerca di sottostringa.
    entity_words = entity_name.lower().split()
    text_lower = text.lower()

    return all(word in text_lower for word in entity_words)


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

    print("Generazione degli esempi di addestramento...")
    for film_data in tqdm(all_films_data, desc="Processando i film"):
        abstract = film_data.get("abstract", "").strip()
        all_triples = film_data.get("triples", [])

        if not abstract or not all_triples:
            continue

        # --- MODIFICA CHIAVE: Filtro delle triple per il task Text2RDF ---
        # Per il task Text2RDF, creiamo una lista di triple "oneste", ovvero
        # solo quelle il cui oggetto è effettivamente menzionato nell'abstract.
        # Questo risolve il problema di chiedere al modello di predire informazioni
        # che non può conoscere (come 'Matt Dallas' in un testo che non lo nomina).
        text2rdf_triples = [
            t for t in all_triples
            if is_entity_in_text(t['object'], abstract)
        ]

        # --- Task 1: Text2RDF ---
        # Generiamo un esempio Text2RDF solo se ci sono triple valide da estrarre.
        if text2rdf_triples:
            input_text1 = f"{abstract} {TEXT_TO_RDF_TOKEN}"
            output_text1 = linearize_triples(text2rdf_triples)
            heavy_tasks_examples.append({"input": input_text1, "output": output_text1})

        # --- Task 2: RDF2Text ---
        # Per questo task, è corretto usare TUTTE le triple, perché vogliamo
        # che il modello impari a generare un testo a partire da un grafo di conoscenza completo.
        input_text2 = f"{linearize_triples(all_triples)} {RDF_TO_TEXT_TOKEN}"
        output_text2 = abstract
        heavy_tasks_examples.append({"input": input_text2, "output": output_text2})

        # --- Task 4: RDF Completion 2 (CONTINUERDF) ---
        # Anche qui, usiamo tutte le triple, poiché il task è la completazione del grafo.
        if len(all_triples) > 1:
            split_point = random.randint(1, len(all_triples) - 1)
            context_triples = all_triples[:split_point]
            completion_triples = all_triples[split_point:]

            input_text4 = f"{linearize_triples(context_triples)} {CONTINUE_RDF_TOKEN}"
            output_text4 = linearize_triples(completion_triples)
            heavy_tasks_examples.append({"input": input_text4, "output": output_text4})

        # --- Raccolta candidati per Task 3 (MLM) ---
        # Questo task si basa su triple individuali, quindi usiamo tutte le triple.
        # La logica originale era già corretta.
        for triple in all_triples:
            # Mascheriamo l'oggetto
            mlm_task_candidates.append({
                "input": (f"{SOT_TOKEN} {SUBJ_TOKEN} {triple['subject']} "
                          f"{PRED_TOKEN} {triple['predicate']} {MASK_TOKEN} {EOT_TOKEN} {MLM_TOKEN}"),
                "output": triple['object']
            })
            # Mascheriamo il predicato
            mlm_task_candidates.append({
                "input": (f"{SOT_TOKEN} {SUBJ_TOKEN} {triple['subject']} "
                          f"{MASK_TOKEN} {OBJ_TOKEN} {triple['object']} {EOT_TOKEN} {MLM_TOKEN}"),
                "output": triple['predicate']
            })

    print(f"Generati {len(heavy_tasks_examples)} esempi per i task generativi.")
    print(f"Generati {len(mlm_task_candidates)} candidati per il task MLM.")

    # --- Bilanciamento e Finalizzazione (questa parte rimane invariata) ---
    print("Bilanciamento del dataset...")
    random.shuffle(mlm_task_candidates)

    # num_heavy_tasks = len(heavy_tasks_examples)
    # selected_mlm_examples = mlm_task_candidates[:num_heavy_tasks]

    # Vogliamo che i task generativi siano circa il 75% del totale.
    # Quindi, il numero di esempi MLM dovrebbe essere circa 1/3 del numero degli altri task.
    num_heavy_tasks = len(heavy_tasks_examples)
    num_mlm_to_select = num_heavy_tasks // 3  # Riduciamo drasticamente gli esempi MLM

    selected_mlm_examples = mlm_task_candidates[:num_mlm_to_select]

    print(f"Selezionati {len(selected_mlm_examples)} esempi MLM per il bilanciamento.")

    final_training_data = heavy_tasks_examples + selected_mlm_examples
    random.shuffle(final_training_data)

    print(f"Dataset finale bilanciato con {len(final_training_data)} esempi totali.")

    # --- Salvataggio su file separati ---
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