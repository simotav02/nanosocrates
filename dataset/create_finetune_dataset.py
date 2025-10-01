# create_finetune_dataset.py

import json
import random
from tqdm import tqdm
import os
import re
from collections import defaultdict, Counter

# --- CONFIGURAZIONE ---
INPUT_JSON_FILE = "film_dataset_10000_final.json"
OUTPUT_DIR = "finetune_all_tasks_data"  # Cartella dedicata per il fine-tuning

# Token speciali
SOT_TOKEN = "<SOT>";
EOT_TOKEN = "<EOT>";
SUBJ_TOKEN = "<SUBJ>"
PRED_TOKEN = "<PRED>";
OBJ_TOKEN = "<OBJ>";
MASK_TOKEN = "<MASK>"
TEXT_TO_RDF_TOKEN = "<Text2RDF>";
RDF_TO_TEXT_TOKEN = "<RDF2Text>"
CONTINUE_RDF_TOKEN = "<CONTINUERDF>";
MLM_TOKEN = "<MLM>"


def linearize_triples(triples):
    if not triples: return ""
    return " ".join(
        [f"{SOT_TOKEN} {SUBJ_TOKEN} {t['subject']} {PRED_TOKEN} {t['predicate']} {OBJ_TOKEN} {t['object']} {EOT_TOKEN}"
         for t in triples])


def is_entity_in_text(entity: str, text: str) -> bool:
    if not isinstance(entity, str): return False
    entity_name = entity.split(':')[-1].replace('_', ' ')
    entity_name = re.sub(r'\(.*?\)', '', entity_name).strip()
    if not entity_name: return False
    return all(word in text.lower() for word in entity_name.lower().split())


def main():
    """
    Legge il JSON, applica regole di pulizia e genera un dataset bilanciato
    con tutti e 4 i task per la fase di fine-tuning.
    """
    print(f"--- Fase 2: Creazione del dataset di Fine-tuning (tutti i task) ---")
    print(f"Caricamento dati da '{INPUT_JSON_FILE}'...")
    with open(INPUT_JSON_FILE, 'r', encoding='utf-8') as f:
        all_films_data = json.load(f)

    print(f"Trovati {len(all_films_data)} record totali nel file JSON.")

    text2rdf_examples, rdf2text_examples, continuerdf_examples, mlm_candidates = [], [], [], []
    records_processed, records_kept = 0, 0
    rejection_reasons = Counter()

    print("Inizio fase di pulizia e generazione degli esempi...")
    for film_data in tqdm(all_films_data, desc="Processando e filtrando i film"):
        records_processed += 1
        title = film_data.get("title", "").strip();
        subject_uri = film_data.get("subject_uri", "").strip()
        abstract = film_data.get("abstract", "").strip();
        all_triples = film_data.get("triples", [])

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

        text2rdf_triples = [t for t in all_triples if is_entity_in_text(t['object'], abstract)]
        if text2rdf_triples:
            text2rdf_examples.append(
                {"input": f"{abstract} {TEXT_TO_RDF_TOKEN}", "output": linearize_triples(text2rdf_triples)})

        rdf2text_examples.append({"input": f"{linearize_triples(all_triples)} {RDF_TO_TEXT_TOKEN}", "output": abstract})

        triples_by_predicate = defaultdict(list)
        for t in all_triples: triples_by_predicate[t['predicate']].append(t)
        for _, triples_group in triples_by_predicate.items():
            if len(triples_group) > 1:
                split_point = random.randint(1, len(triples_group) - 1)
                context, completion = triples_group[:split_point], triples_group[split_point:]
                continuerdf_examples.append({"input": f"{linearize_triples(context)} {CONTINUE_RDF_TOKEN}",
                                             "output": linearize_triples(completion)})

        for triple in all_triples:
            mlm_candidates.append({
                                      "input": f"{SOT_TOKEN} {SUBJ_TOKEN} {triple['subject']} {PRED_TOKEN} {triple['predicate']} {MASK_TOKEN} {EOT_TOKEN} {MLM_TOKEN}",
                                      "output": triple['object']})
            mlm_candidates.append({
                                      "input": f"{SOT_TOKEN} {SUBJ_TOKEN} {triple['subject']} {MASK_TOKEN} {OBJ_TOKEN} {triple['object']} {EOT_TOKEN} {MLM_TOKEN}",
                                      "output": triple['predicate']})

    print(
        "\n" + "=" * 50 + "\n--- Risultati della Fase di Pulizia dei Dati ---\n" + f"Record totali processati: {records_processed}\n" + f"Record scartati: {records_processed - records_kept} ({((records_processed - records_kept) / records_processed) * 100:.2f}%)\n" + f"Record tenuti per il training: {records_kept}\n\nDettaglio motivi di scarto:")
    for reason, count in rejection_reasons.items(): print(f"- {reason}: {count}")
    print("=" * 50 + "\n")

    print("\n--- Bilanciamento Finale a Percentuali Fisse ---")
    random.shuffle(text2rdf_examples);
    random.shuffle(rdf2text_examples);
    random.shuffle(continuerdf_examples);
    random.shuffle(mlm_candidates)

    TARGET_PERCENTAGES = {'Text2RDF': 0.25, 'RDF2Text': 0.25, 'CONTINUERDF': 0.30, 'MLM': 0.20}
    available_examples = {'Text2RDF': text2rdf_examples, 'RDF2Text': rdf2text_examples,
                          'CONTINUERDF': continuerdf_examples, 'MLM': mlm_candidates}

    base_task_count = min(len(available_examples['Text2RDF']), len(available_examples['RDF2Text']))
    if base_task_count == 0:
        print("ERRORE: Text2RDF o RDF2Text non hanno prodotto esempi.");
        return

    base_size = int(base_task_count / TARGET_PERCENTAGES['Text2RDF'])
    print(f"Task di riferimento ha {base_task_count} esempi. Dimensione target del dataset: ~{base_size}")

    final_data = []
    for task_name, percentage in TARGET_PERCENTAGES.items():
        num_to_sample = int(base_size * percentage)
        candidates = available_examples[task_name]
        num_to_sample = min(num_to_sample, len(candidates))
        final_data.extend(candidates[:num_to_sample])

    random.shuffle(final_data)

    final_counts = Counter(t.split(' ')[-1] for t in [item['input'] for item in final_data])
    total_examples = len(final_data)

    print("\n--- Statistiche Dataset Finale Bilanciato ---")
    print(f"TOTALE ESEMPI: {total_examples}")
    tasks_in_order = ['Text2RDF', 'RDF2Text', 'CONTINUERDF', 'MLM']
    for task in tasks_in_order:
        count = final_counts[f"<{task}>"]
        percentage = (count / total_examples) * 100 if total_examples > 0 else 0
        print(f"- Task '{task}': {count} esempi ({percentage:.2f}%)")
    print("=" * 50 + "\n")

    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    source_filepath = os.path.join(OUTPUT_DIR, "train.source")
    target_filepath = os.path.join(OUTPUT_DIR, "train.target")

    print(f"Salvataggio del corpus di fine-tuning in '{source_filepath}' e '{target_filepath}'...")
    with open(source_filepath, 'w', encoding='utf-8') as f_source, open(target_filepath, 'w',
                                                                        encoding='utf-8') as f_target:
        for example in tqdm(final_data, desc="Scrivendo i file"):
            f_source.write(example["input"] + "\n")
            f_target.write(example["output"] + "\n")

    print("\nPROCESSO COMPLETATO.")
    print(f"File di training generati con successo nella nuova cartella '{OUTPUT_DIR}'.")


if __name__ == "__main__":
    main()