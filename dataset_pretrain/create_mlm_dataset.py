# create_mlm_dataset.py
import json
import os
from tqdm import tqdm
import random

# --- CONFIGURAZIONE ---
# Assicurati che il nome del file JSON corrisponda a quello generato da data_creation_2.py
INPUT_JSON_FILE = "../dataset/film_dataset_5000_cleaned.json"
OUTPUT_DIR = "mlm_only_data"

# Token speciali
SOT_TOKEN, EOT_TOKEN = "<SOT>", "<EOT>"
SUBJ_TOKEN, PRED_TOKEN, OBJ_TOKEN = "<SUBJ>", "<PRED>", "<OBJ>"
MASK_TOKEN, MLM_TOKEN = "<MASK>", "<MLM>"


def main():
    """
    Crea un dataset contenente ESCLUSIVAMENTE esempi per il task MLM
    per la fase di Task-Adaptation.
    """
    if not os.path.exists(INPUT_JSON_FILE):
        print(f"ERRORE: File '{INPUT_JSON_FILE}' non trovato. Esegui prima data_creation_2.py.")
        return

    print(f"Caricamento dati da '{INPUT_JSON_FILE}'...")
    with open(INPUT_JSON_FILE, 'r', encoding='utf-8') as f:
        all_films_data = json.load(f)

    mlm_examples = []
    for film_data in tqdm(all_films_data, desc="Generando solo esempi MLM"):
        for triple in film_data.get("triples", []):
            if not all(k in triple for k in ['subject', 'predicate', 'object']):
                continue  # Salta triple malformate

            # Mascherare l'oggetto
            mlm_examples.append({
                "input": f"{SOT_TOKEN} {SUBJ_TOKEN} {triple['subject']} {PRED_TOKEN} {triple['predicate']} {MASK_TOKEN} {EOT_TOKEN} {MLM_TOKEN}",
                "output": triple['object']
            })
            # Mascherare il predicato
            mlm_examples.append({
                "input": f"{SOT_TOKEN} {SUBJ_TOKEN} {triple['subject']} {MASK_TOKEN} {OBJ_TOKEN} {triple['object']} {EOT_TOKEN} {MLM_TOKEN}",
                "output": triple['predicate']
            })

    random.shuffle(mlm_examples)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    source_path = os.path.join(OUTPUT_DIR, "train.source")
    target_path = os.path.join(OUTPUT_DIR, "train.target")

    print(f"Salvataggio di {len(mlm_examples)} esempi in '{OUTPUT_DIR}'...")
    with open(source_path, 'w', encoding='utf-8') as f_src, open(target_path, 'w', encoding='utf-8') as f_tgt:
        for ex in mlm_examples:
            f_src.write(ex['input'] + '\n')
            f_tgt.write(ex['output'] + '\n')

    print("Processo completato.")


if __name__ == "__main__":
    main()