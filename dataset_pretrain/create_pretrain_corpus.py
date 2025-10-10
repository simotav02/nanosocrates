import json
import os
from tqdm import tqdm
import random

INPUT_JSON_FILE = "../dataset/film_dataset_5000_cleaned.json"
OUTPUT_DIR = "pretrain_corpus_data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "pretrain_corpus.txt")

SOT_TOKEN, EOT_TOKEN = "<SOT>", "<EOT>"
SUBJ_TOKEN, PRED_TOKEN, OBJ_TOKEN = "<SUBJ>", "<PRED>", "<OBJ>"


def linearize_triples_raw(triples):
    """Linearizza le triple in un formato grezzo, senza token di task."""
    if not triples:
        return ""
    return " ".join(
        [f"{SOT_TOKEN} {SUBJ_TOKEN} {t['subject']} {PRED_TOKEN} {t['predicate']} {OBJ_TOKEN} {t['object']} {EOT_TOKEN}"
         for t in triples]
    )


def main():
    """
    Crea un corpus di testo puro (agnostico ai task) per il pre-training,
    unendo abstract e triple linearizzate grezze.
    """
    if not os.path.exists(INPUT_JSON_FILE):
        print(f"ERRORE: File '{INPUT_JSON_FILE}' non trovato. Esegui prima data_creation_2.py.")
        return

    print(f"Caricamento dati da '{INPUT_JSON_FILE}'...")
    with open(INPUT_JSON_FILE, 'r', encoding='utf-8') as f:
        all_films_data = json.load(f)

    corpus_lines = []
    print("Estrazione e combinazione di abstract e triple...")
    for film_data in tqdm(all_films_data, desc="Creando il corpus di pre-training"):
        abstract = film_data.get("abstract", "").strip()
        triples = film_data.get("triples", [])

        if abstract:
            corpus_lines.append(abstract)

        if triples:
            linearized_rdf = linearize_triples_raw(triples)
            if linearized_rdf:
                corpus_lines.append(linearized_rdf)

    random.shuffle(corpus_lines)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Salvataggio di {len(corpus_lines)} righe di testo puro in '{OUTPUT_FILE}'...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for line in corpus_lines:
            f.write(line + '\n')

    print("\n--- PROCESSO COMPLETATO ---")
    print(f"Corpus di pre-training puro creato con successo in '{OUTPUT_FILE}'.")
    print("Ora puoi usare questo file come input per lo script 'pretrain_dateset_T5.py'.")


if __name__ == "__main__":
    main()