import json
import random
from tqdm import tqdm

# --- CONFIGURAZIONE ---
INPUT_JSON_FILE = "film_dataset_1000.json"
OUTPUT_CORPUS_FILE = "training_corpus_1000.txt"

# Definiamo i token speciali come costanti per evitare errori di battitura
# e per coerenza con la traccia del progetto.
SOT_TOKEN = "<SOT>"  # Start of Triple
EOT_TOKEN = "<EOT>"  # End of Triple
SUBJ_TOKEN = "<SUBJ>"
PRED_TOKEN = "<PRED>"
OBJ_TOKEN = "<OBJ>"
MASK_TOKEN = "<MASK>"

TEXT_TO_RDF_TOKEN = "<Text2RDF>"
RDF_TO_TEXT_TOKEN = "<RDF2Text>"
CONTINUE_RDF_TOKEN = "<CONTINUERDF>"


def linearize_triples(triples):
    """
    Converte una lista di triple in una singola stringa serializzata.
    Input: [{"subject": "dbr:A", "predicate": "dbo:B", "object": "dbr:C"}, ...]
    Output: "<SOT> <SUBJ> dbr:A <PRED> dbo:B <OBJ> dbr:C <EOT> ..."
    """
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
    Legge il dataset JSON e genera le coppie (input, output) per tutti i task.
    """
    print(f"Caricamento dati da '{input_filepath}'...")
    with open(input_filepath, 'r', encoding='utf-8') as f:
        all_films_data = json.load(f)

    training_examples = []
    print("Generazione degli esempi di addestramento per i 4 task...")

    for film_data in tqdm(all_films_data, desc="Processando i film"):
        abstract = film_data.get("abstract", "").strip()
        triples = film_data.get("triples", [])

        if not abstract or not triples:
            continue

        # --- Task 1: Text2RDF ---
        input_text1 = f"{abstract} {TEXT_TO_RDF_TOKEN}"
        output_text1 = linearize_triples(triples)
        training_examples.append({"input": input_text1, "output": output_text1})

        # --- Task 2: RDF2Text ---
        input_text2 = f"{linearize_triples(triples)} {RDF_TO_TEXT_TOKEN}"
        output_text2 = abstract
        training_examples.append({"input": input_text2, "output": output_text2})

        # --- Task 3: RDF Completion 1 (Masked Language Modeling) ---
        for triple in triples:
            # Mascheriamo l'oggetto
            input_text3_obj = (f"{SOT_TOKEN} {SUBJ_TOKEN} {triple['subject']} "
                               f"{PRED_TOKEN} {triple['predicate']} "
                               f"{OBJ_TOKEN} {MASK_TOKEN} {EOT_TOKEN}")
            output_text3_obj = triple['object']
            training_examples.append({"input": input_text3_obj, "output": output_text3_obj})

            # Mascheriamo il predicato
            input_text3_pred = (f"{SOT_TOKEN} {SUBJ_TOKEN} {triple['subject']} "
                                f"{PRED_TOKEN} {MASK_TOKEN} "
                                f"{OBJ_TOKEN} {triple['object']} {EOT_TOKEN}")
            output_text3_pred = triple['predicate']
            training_examples.append({"input": input_text3_pred, "output": output_text3_pred})

        # --- Task 4: RDF Completion 2 (RDF Generation) ---
        if len(triples) > 1:
            # Usiamo la prima tripla come contesto e le restanti come target
            # Si potrebbero usare strategie più complesse (es. split casuali),
            # ma questa è una base di partenza solida.
            context_triples = [triples[0]]
            completion_triples = triples[1:]

            input_text4 = f"{linearize_triples(context_triples)} {CONTINUE_RDF_TOKEN}"
            output_text4 = linearize_triples(completion_triples)

            if input_text4 and output_text4:
                training_examples.append({"input": input_text4, "output": output_text4})

    return training_examples


def main():
    """
    Funzione principale che orchestra la creazione del corpus di addestramento.
    """
    # Genera tutte le coppie di addestramento
    training_data = generate_training_data(INPUT_JSON_FILE)

    if not training_data:
        print("Nessun dato di addestramento generato. Controlla il file di input.")
        return

    print(f"\nGenerati un totale di {len(training_data)} esempi di addestramento.")

    # Mescola gli esempi per assicurare che il modello non veda i task in blocchi
    print("Mescolamento degli esempi...")
    random.shuffle(training_data)

    # Scrive il corpus finale in formato text-to-text (input \t output)
    print(f"Salvataggio del corpus in '{OUTPUT_CORPUS_FILE}'...")
    try:
        with open(OUTPUT_CORPUS_FILE, 'w', encoding='utf-8') as f:
            for example in tqdm(training_data, desc="Scrivendo su file"):
                # Rimuoviamo eventuali newline dall'input/output per mantenere una riga per esempio
                clean_input = example['input'].replace('\n', ' ').strip()
                clean_output = example['output'].replace('\n', ' ').strip()
                f.write(f"{clean_input}\t{clean_output}\n")

        print("\nPROCESSO COMPLETATO.")
        print(f"Il corpus di addestramento unificato è stato salvato con successo.")
    except IOError as e:
        print(f"Errore durante il salvataggio del file: {e}")


if __name__ == "__main__":
    main()