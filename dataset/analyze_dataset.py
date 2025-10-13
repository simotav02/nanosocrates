import json

DATASET_FILE = "film_dataset_10000_cleaned.json"

WHITELISTED_PREDICATES = {
    "dbo:director", "dbo:writer", "dbo:starring", "dbo:producer",
    "dbo:musicComposer", "dbo:country", "dbo:language", "dbo:releaseDate",
    "dbo:distributor", "dbo:cinematography", "dbo:editing", "dbo:imdbId",
    "rdf:type", "rdfs:label"
}


def validate_dataset(filepath):
    """
    Esegue una serie di controlli programmatici su un dataset JSON.
    """
    print(f"--- Inizio validazione di '{filepath}' ---")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print("✅ Successo: il file è un JSON valido e leggibile.")
    except json.JSONDecodeError as e:
        print(f"❌ Errore Critico: Il file non è un JSON valido. Errore: {e}")
        return
    except FileNotFoundError:
        print(f"❌ Errore Critico: File non trovato: '{filepath}'")
        return

    if not isinstance(data, list):
        print("❌ Errore Strutturale: Il JSON principale non è una lista.")
        return

    print(f"Trovati {len(data)} record nel dataset.")

    all_predicates_found = set()
    errors = []

    for i, record in enumerate(data):
        expected_keys = {"title", "subject_uri", "abstract", "triples"}
        if not isinstance(record, dict) or set(record.keys()) != expected_keys:
            errors.append(f"Record #{i}: Schema non corretto. Trovate chiavi: {list(record.keys())}")
            continue

        if not record.get("abstract"):
            errors.append(f"Record #{i} ('{record.get('title')}'): Abstract vuoto.")
        if not record.get("triples"):
            errors.append(f"Record #{i} ('{record.get('title')}'): Lista triple vuota.")

        for triple in record["triples"]:
            expected_triple_keys = {"subject", "predicate", "object"}
            if not isinstance(triple, dict) or set(triple.keys()) != expected_triple_keys:
                errors.append(f"Record #{i}: Tripla con schema non corretto: {triple}")
                continue

            all_predicates_found.add(triple['predicate'])

    if errors:
        print(f"\n❌ Trovati {len(errors)} errori di validazione:")
        for error in errors[:10]:
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... e altri {len(errors) - 10} errori.")
    else:
        print("✅ Successo: Tutti i record hanno lo schema corretto e non hanno campi critici vuoti.")

    print("\n--- Controllo Vocabolario Predicati ---")
    unauthorized_predicates = all_predicates_found - WHITELISTED_PREDICATES

    if unauthorized_predicates:
        print(f"❌ Errore: Trovati {len(unauthorized_predicates)} predicati non presenti nella whitelist:")
        print(unauthorized_predicates)
    else:
        print("✅ Successo: Tutti i predicati nel dataset sono conformi alla whitelist.")
        print("Predicati trovati:", sorted(list(all_predicates_found)))

    print("\n--- Validazione completata ---")


if __name__ == "__main__":
    validate_dataset(DATASET_FILE)