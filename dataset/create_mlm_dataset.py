# create_mlm_dataset.py

import json
import random
from tqdm import tqdm
import os
import re
from collections import Counter

# --- CONFIGURAZIONE ---
INPUT_JSON_FILE = "film_dataset_10000_final.json"
OUTPUT_DIR = "pretrain_mlm_data"  # Cartella dedicata per il pre-training

# Token speciali
SOT_TOKEN = "<SOT>";
EOT_TOKEN = "<EOT>";
SUBJ_TOKEN = "<SUBJ>"
PRED_TOKEN = "<PRED>";
OBJ_TOKEN = "<OBJ>";
MASK_TOKEN = "<MASK>"
MLM_TOKEN = "<MLM>"


def main():
    """
    Legge il dataset JSON pulito e genera un corpus di training
    contenente ESCLUSIVAMENTE esempi per il task Masked Language Modeling (MLM).
    """
    print(f"--- Fase 1: Creazione del dataset di Pre-training MLM ---")
    print(f"Caricamento dati da '{INPUT_JSON_FILE}'...")
    with open(INPUT_JSON_FILE, 'r', encoding='utf-8') as f:
        all_films_data = json.load(f)

    print(f"Trovati {len(all_films_data)} record totali nel file JSON.")

    mlm_examples = []
    records_processed, records_kept = 0, 0
    rejection_reasons = Counter()

    print("Inizio fase di pulizia e generazione degli esempi MLM...")
    for film_data in tqdm(all_films_data, desc="Processando e filtrando i film"):
        records_processed += 1

        title = film_data.get("title", "").strip()
        subject_uri = film_data.get("subject_uri", "").strip()
        abstract = film_data.get("abstract", "").strip()
        all_triples = film_data.get("triples", [])

        # --- BLOCCO DI VALIDAZIONE E PULIZIA DEL RECORD (identico per entrambe le fasi) ---
        if not all([title, subject_uri, abstract, all_triples]):
            rejection_reasons['dati_mancanti'] += 1;
            continue
        all_triples = [t for t in all_triples if t.get('subject') == subject_uri]
        if not all_triples:
            rejection_reasons['nessuna_tripla_coerente'] += 1;
            continue
        predicates_in_record = {t['predicate'] for t in all_triples}
        if not {"dbo:director", "dbo:starring"}.issubset(predicates_in_record):
            rejection_reasons['triple_incomplete'] += 1;
            continue
        if len(abstract) < 250:
            rejection_reasons['abstract_troppo_corto'] += 1;
            continue
        title_cleaned = title.split('(')[0].strip()
        if not title_cleaned or title_cleaned.lower() not in abstract.lower():
            rejection_reasons['titolo_non_in_abstract'] += 1;
            continue

        records_kept += 1

        # --- GENERAZIONE ESCLUSIVA DI ESEMPI MLM ---
        for triple in all_triples:
            # Maschera Oggetto
            mlm_examples.append({
                                    "input": f"{SOT_TOKEN} {SUBJ_TOKEN} {triple['subject']} {PRED_TOKEN} {triple['predicate']} {MASK_TOKEN} {EOT_TOKEN} {MLM_TOKEN}",
                                    "output": triple['object']})
            # Maschera Predicato
            mlm_examples.append({
                                    "input": f"{SOT_TOKEN} {SUBJ_TOKEN} {triple['subject']} {MASK_TOKEN} {OBJ_TOKEN} {triple['object']} {EOT_TOKEN} {MLM_TOKEN}",
                                    "output": triple['predicate']})

    # --- STAMPA DIAGNOSTICA PULIZIA ---
    print(
        "\n" + "=" * 50 + "\n--- Risultati della Fase di Pulizia dei Dati ---\n" + f"Record totali processati: {records_processed}\n" + f"Record scartati: {records_processed - records_kept} ({((records_processed - records_kept) / records_processed) * 100:.2f}%)\n" + f"Record tenuti per il training: {records_kept}\n\nDettaglio motivi di scarto:")
    for reason, count in rejection_reasons.items(): print(f"- {reason}: {count}")
    print("=" * 50 + "\n")

    random.shuffle(mlm_examples)
    print(f"Generati {len(mlm_examples)} esempi totali per il pre-training MLM.")

    # --- Salvataggio su file ---
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    source_filepath = os.path.join(OUTPUT_DIR, "train.source")
    target_filepath = os.path.join(OUTPUT_DIR, "train.target")

    print(f"Salvataggio del corpus di pre-training in '{source_filepath}' e '{target_filepath}'...")
    with open(source_filepath, 'w', encoding='utf-8') as f_source, open(target_filepath, 'w',
                                                                        encoding='utf-8') as f_target:
        for example in tqdm(mlm_examples, desc="Scrivendo i file"):
            f_source.write(example["input"] + "\n")
            f_target.write(example["output"] + "\n")

    print("\nPROCESSO COMPLETATO.")
    print(f"Dataset di pre-training MLM generato con successo nella cartella '{OUTPUT_DIR}'.")


if __name__ == "__main__":
    main()