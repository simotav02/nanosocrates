import json
import random
from tqdm import tqdm

# --- CONFIGURAZIONE ---
INPUT_JSON_FILE = "film_dataset.json"
OUTPUT_CORPUS_FILE = "training_corpus.txt"

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


def generate_training_data(input_filepath):
    """
    Legge il dataset JSON e genera le coppie (input, output)
    con una strategia di bilanciamento dei task.
    """
    print(f"Caricamento dati da '{input_filepath}'...")
    with open(input_filepath, 'r', encoding='utf-8') as f:
        all_films_data = json.load(f)

    training_examples = []
    print("Generazione degli esempi di addestramento bilanciati...")

    for film_data in tqdm(all_films_data, desc="Processando i film"):
        abstract = film_data.get("abstract", "").strip()
        triples = film_data.get("triples", [])

        if not abstract or not triples:
            continue

        # --- Task 1: Text2RDF (1 esempio per film) ---
        input_text1 = f"{abstract} {TEXT_TO_RDF_TOKEN}"
        output_text1 = linearize_triples(triples)
        training_examples.append({"input": input_text1, "output": output_text1})

        # --- Task 2: RDF2Text (1 esempio per film) ---
        input_text2 = f"{linearize_triples(triples)} {RDF_TO_TEXT_TOKEN}"
        output_text2 = abstract
        training_examples.append({"input": input_text2, "output": output_text2})

        # --- Task 4: RDF Completion 2 (1 esempio per film, se possibile) ---
        if len(triples) > 1:
            # Scegli un punto di divisione casuale per rendere il task più vario
            split_point = random.randint(1, len(triples) - 1)
            context_triples = triples[:split_point]
            completion_triples = triples[split_point:]

            input_text4 = f"{linearize_triples(context_triples)} {CONTINUE_RDF_TOKEN}"
            output_text4 = linearize_triples(completion_triples)

            if input_text4 and output_text4:
                training_examples.append({"input": input_text4, "output": output_text4})

        # Generiamo un numero fisso e piccolo per ogni film, per evitare che il task MLM domini il dataset.
        # Stabiliamo un numero di esempi MLM da generare per film.
        num_mlm_samples_per_film = 1

        # Creiamo una lista di tutti i possibili esempi MLM per questo film
        possible_mlm_examples = []
        for triple in triples:
            # Mascheriamo l'oggetto
            possible_mlm_examples.append({
                "input": (f"{SOT_TOKEN} {SUBJ_TOKEN} {triple['subject']} "
                          f"{PRED_TOKEN} {triple['predicate']} "
                          f"{OBJ_TOKEN} {MASK_TOKEN} {EOT_TOKEN}"),
                "output": triple['object']
            })
            # Mascheriamo il predicato
            possible_mlm_examples.append({
                "input": (f"{SOT_TOKEN} {SUBJ_TOKEN} {triple['subject']} "
                          f"{PRED_TOKEN} {MASK_TOKEN} "
                          f"{OBJ_TOKEN} {triple['object']} {EOT_TOKEN}"),
                "output": triple['predicate']
            })

        # Mescoliamo e campioniamo un sottoinsieme di questi esempi
        random.shuffle(possible_mlm_examples)

        # Aggiungiamo al dataset finale un numero di esempi MLM
        # pari a min(num_mlm_samples_per_film, len(possible_mlm_examples))
        # per gestire film con poche triple.
        for i in range(min(num_mlm_samples_per_film, len(possible_mlm_examples))):
            training_examples.append(possible_mlm_examples[i])

    return training_examples


def main():
    """Funzione principale che orchestra la creazione del corpus di addestramento."""
    training_data = generate_training_data(INPUT_JSON_FILE)

    if not training_data:
        print("Nessun dato di addestramento generato.")
        return

    print(f"\nGenerati un totale di {len(training_data)} esempi di addestramento.")
    print("Mescolamento degli esempi...")
    random.shuffle(training_data)

    print(f"Salvataggio del corpus in '{OUTPUT_CORPUS_FILE}'...")
    try:
        with open(OUTPUT_CORPUS_FILE, 'w', encoding='utf-8') as f:
            for example in tqdm(training_data, desc="Scrivendo su file"):
                clean_input = example['input'].replace('\n', ' ').strip()
                clean_output = example['output'].replace('\n', ' ').strip()
                f.write(f"{clean_input}\t{clean_output}\n")
        print("\nPROCESSO COMPLETATO.")
        print(f"Il corpus di addestramento bilanciato è stato salvato con successo.")
    except IOError as e:
        print(f"Errore durante il salvataggio del file: {e}")


if __name__ == "__main__":
    main()