import json
import os
from tqdm import tqdm
import random
from collections import Counter

INPUT_JSON_FILE = "../dataset/film_dataset_30000_cleaned.json"
OUTPUT_DIR = "pretrain_corpus_data_v3"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "pretrain_corpus.txt")

MIN_WORDS_THRESHOLD = 15
MAX_WORDS_THRESHOLD = 250
TARGET_BALANCE_RATIO = 0.5

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
    Crea un corpus di testo puro di alta qualità per il pre-training,
    filtrando, de-duplicando e bilanciando i dati.
    """
    if not os.path.exists(INPUT_JSON_FILE):
        print(f"ERRORE: File '{INPUT_JSON_FILE}' non trovato. Esegui prima data_creation_2.py.")
        return

    print(f"--- Creazione Corpus di Pre-Training di Alta Qualità (v2) ---")
    print(f"Caricamento dati da '{INPUT_JSON_FILE}'...")
    with open(INPUT_JSON_FILE, 'r', encoding='utf-8') as f:
        all_films_data = json.load(f)

    print("\n1/4 - Estrazione e filtraggio iniziale del corpus...")

    raw_abstracts = []
    raw_rdf_chunks = []
    stats = Counter()

    for film_data in tqdm(all_films_data, desc="Processando i film"):
        stats['total_films_processed'] += 1

        abstract = film_data.get("abstract", "").strip()
        triples = film_data.get("triples", [])

        if abstract:
            stats['abstracts_found'] += 1
            word_count = len(abstract.split())
            if MIN_WORDS_THRESHOLD <= word_count <= MAX_WORDS_THRESHOLD:
                raw_abstracts.append(abstract)
                stats['abstracts_kept_after_filtering'] += 1
            else:
                stats['abstracts_rejected_by_length'] += 1

        if triples:
            stats['rdf_chunks_found'] += 1
            linearized_rdf = linearize_triples_raw(triples)
            if linearized_rdf:
                word_count = len(linearized_rdf.split())
                if MIN_WORDS_THRESHOLD <= word_count <= MAX_WORDS_THRESHOLD:
                    raw_rdf_chunks.append(linearized_rdf)
                    stats['rdf_chunks_kept_after_filtering'] += 1
                else:
                    stats['rdf_chunks_rejected_by_length'] += 1

    print("\n2/4 - Rimozione dei duplicati...")

    unique_abstracts = sorted(list(set(raw_abstracts)))
    unique_rdf_chunks = sorted(list(set(raw_rdf_chunks)))

    stats['abstracts_after_deduplication'] = len(unique_abstracts)
    stats['rdf_chunks_after_deduplication'] = len(unique_rdf_chunks)

    print("\n3/4 - Bilanciamento del corpus (target 50% abstract, 50% RDF)...")

    final_corpus_lines = []

    if not unique_abstracts or not unique_rdf_chunks:
        print(
            "ATTENZIONE: Una delle due fonti di dati (abstract o RDF) è vuota dopo il filtraggio. Il bilanciamento non è possibile.")
        final_corpus_lines = unique_abstracts + unique_rdf_chunks
    else:
        min_category_size = min(len(unique_abstracts), len(unique_rdf_chunks))

        stats['balancing_base_size'] = min_category_size

        random.shuffle(unique_abstracts)
        random.shuffle(unique_rdf_chunks)

        balanced_abstracts = unique_abstracts[:min_category_size]
        balanced_rdf_chunks = unique_rdf_chunks[:min_category_size]

        final_corpus_lines.extend(balanced_abstracts)
        final_corpus_lines.extend(balanced_rdf_chunks)

    random.shuffle(final_corpus_lines)

    print("\n--- Statistiche del Processo di Creazione Corpus ---")
    print(f"Film totali processati: {stats['total_films_processed']:,}")
    print("-" * 40)
    print("Abstracts:")
    print(f"  - Trovati: {stats['abstracts_found']:,}")
    print(
        f"  - Scartati per lunghezza (<{MIN_WORDS_THRESHOLD} or >{MAX_WORDS_THRESHOLD} parole): {stats['abstracts_rejected_by_length']:,}")
    print(f"  - Mantenuti dopo filtraggio: {stats['abstracts_kept_after_filtering']:,}")
    print(f"  - Mantenuti dopo de-duplicazione: {stats['abstracts_after_deduplication']:,}")
    print("-" * 40)
    print("Chunk RDF:")
    print(f"  - Trovati: {stats['rdf_chunks_found']:,}")
    print(f"  - Scartati per lunghezza: {stats['rdf_chunks_rejected_by_length']:,}")
    print(f"  - Mantenuti dopo filtraggio: {stats['rdf_chunks_kept_after_filtering']:,}")
    print(f"  - Mantenuti dopo de-duplicazione: {stats['rdf_chunks_after_deduplication']:,}")
    print("-" * 40)
    print("Bilanciamento:")
    print(f"  - Dimensione di riferimento (categoria più piccola): {stats.get('balancing_base_size', 'N/A'):,}")
    print(f"  - Righe totali nel corpus finale: {len(final_corpus_lines):,}")
    print("=" * 50)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n4/4 - Salvataggio di {len(final_corpus_lines)} righe di testo pulito e bilanciato in '{OUTPUT_FILE}'...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for line in tqdm(final_corpus_lines, desc="Scrivendo il file"):
            f.write(line + '\n')

    print("\n--- PROCESSO COMPLETATO ---")
    print(f"Corpus di pre-training di alta qualità creato con successo in '{OUTPUT_FILE}'.")
    print("Ora puoi usare questo file come input per lo script 'pretrain_dateset_T5.py'.")


if __name__ == "__main__":
    main()
