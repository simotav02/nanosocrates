import json
from SPARQLWrapper import SPARQLWrapper, JSON
from collections import defaultdict
import time
from tqdm import tqdm

# --- Configuration ---
DBPEDIA_SPARQL_ENDPOINT = "https://dbpedia.org/sparql"
FILM_LIMIT = 30000
PAGE_SIZE = 1000
OUTPUT_FILE = f"film_dataset_{FILM_LIMIT}_cleaned.json"

# A mapping to convert full URIs to shortened, prefixed versions (e.g., dbr:Casablanca).
PREFIX_MAP = {
    "http://dbpedia.org/resource/": "dbr:",
    "http://dbpedia.org/ontology/": "dbo:",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#": "rdf:",
    "http://www.w3.org/2000/01/rdf-schema#": "rdfs:"
}

# A whitelist of predicates to keep, focusing on relevant film metadata.
WHITELISTED_PREDICATES = [
    "dbo:director", "dbo:writer", "dbo:starring", "dbo:producer",
    "dbo:musicComposer", "dbo:country", "dbo:language", "dbo:releaseDate",
    "dbo:distributor", "dbo:cinematography", "dbo:editing", "dbo:imdbId",
    "rdf:type", "rdfs:label"
]


def shorten_uri(uri):
    """Converts a full URI string to its prefixed version if a mapping exists."""
    if not isinstance(uri, str):
        return uri
    for base, prefix in PREFIX_MAP.items():
        if uri.startswith(base):
            return uri.replace(base, prefix, 1)
    return uri


def fetch_and_process_film_data(endpoint_url, total_limit, page_size):
    """
    Fetches film data from a SPARQL endpoint using pagination.
    It retrieves triples and abstracts, filters them, and groups them by film.
    """
    sparql = SPARQLWrapper(endpoint_url)
    predicate_filter_string = ", ".join(WHITELISTED_PREDICATES)

    # Use a defaultdict to easily group triples and metadata by film URI.
    films_data = defaultdict(lambda: {"triples": set(), "abstract": None, "title": None})
    num_pages = (total_limit + page_size - 1) // page_size

    for i in range(num_pages):
        offset = i * page_size
        print(f"--- Esecuzione query pagina {i + 1}/{num_pages} (OFFSET {offset}, LIMIT {page_size}) ---")

        # This SPARQL query fetches films in pages, then gets all their relevant triples.
        query = f"""
            PREFIX dbo: <http://dbpedia.org/ontology/>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

            SELECT ?film ?filmLabel ?abstract ?p ?o
            WHERE {{
                # Subquery to get a stable page of films, avoiding memory issues.
                {{
                    SELECT DISTINCT ?film WHERE {{ ?film a dbo:Film . }} ORDER BY ?film LIMIT {page_size} OFFSET {offset}
                }}

                ?film ?p ?o .

                # Keep URIs, or literals that are in English or have no specified language.
                FILTER(!isLiteral(?o) || lang(?o) = 'en' || lang(?o) = "")

                # Filter to only include predicates from our whitelist.
                FILTER(?p IN ({predicate_filter_string}))

                # Optionally fetch the English label and abstract for each film.
                OPTIONAL {{ ?film rdfs:label ?filmLabel . FILTER(LANG(?filmLabel) = 'en') }}
                OPTIONAL {{ ?film dbo:abstract ?abstract . FILTER(LANG(?abstract) = 'en') }}
            }}
        """

        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)

        try:
            results = sparql.query().convert()["results"]["bindings"]
            print(f"Query completata. Ricevute {len(results)} righe.")
            if not results and offset > 0:
                print("Nessun altro risultato da DBpedia. Interruzione.")
                break
        except Exception as e:
            # Basic retry mechanism for network errors.
            print(f"Errore durante la query SPARQL (offset {offset}): {e}")
            print("Attendo 10 secondi e riprovo...")
            time.sleep(10)
            continue

        # Process the results, grouping them by film URI.
        for res in tqdm(results, desc=f"Processando risultati pagina {i + 1}"):
            film_uri = res['film']['value']

            # Populate title and abstract (once per film).
            if not films_data[film_uri]['title'] and 'filmLabel' in res:
                films_data[film_uri]['title'] = res['filmLabel']['value']
            if not films_data[film_uri]['abstract'] and 'abstract' in res:
                films_data[film_uri]['abstract'] = res['abstract']['value'].split('\n')[0].strip()

            predicate_val = res['p']['value']
            object_val = res['o']['value']

            # Special filter for rdf:type to keep only ontology-related types.
            if predicate_val == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type":
                if not object_val.startswith("http://dbpedia.org/ontology/"):
                    continue

            # Add the shortened triple to the set for the current film.
            films_data[film_uri]['triples'].add((
                shorten_uri(film_uri),
                shorten_uri(predicate_val),
                shorten_uri(object_val)
            ))

        time.sleep(1)  # Be polite to the SPARQL endpoint.

    # Final cleaning and formatting phase.
    print("\n--- Fase di pulizia e formattazione finale ---")
    curated_data = []
    for uri, data in tqdm(films_data.items(), desc="Finalizzando i record"):
        # Discard films that are missing an abstract or useful triples.
        if not data['abstract'] or not data['triples']:
            continue

        # Use the URI to create a fallback title if the rdfs:label is missing.
        title = data['title'] if data['title'] else uri.split('/')[-1].replace('_', ' ')

        # Convert the set of tuples into a sorted list of dictionaries for clean JSON output.
        curated_data.append({
            "title": title,
            "subject_uri": shorten_uri(uri),
            "abstract": data['abstract'],
            "triples": [{"subject": s, "predicate": p, "object": o} for s, p, o in sorted(list(data['triples']))]
        })

    return curated_data


def main():
    """
    Main function to orchestrate the data acquisition process and save the results.
    """
    print(f"--- Inizio acquisizione dati per {FILM_LIMIT} film ---")
    final_data = fetch_and_process_film_data(DBPEDIA_SPARQL_ENDPOINT, total_limit=FILM_LIMIT, page_size=PAGE_SIZE)

    if not final_data:
        print("ERRORE: Nessun dato recuperato. Controlla la connessione o l'endpoint SPARQL. Uscita.")
        return

    # Save the final list of curated film data to a JSON file.
    print(f"\n--- Salvataggio dei dati in '{OUTPUT_FILE}' ---")
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, ensure_ascii=False, indent=4)
        print(f"PROCESSO COMPLETATO.")
        print(f"Salvati i dati di {len(final_data)} film.")
    except Exception as e:
        print(f"Errore durante il salvataggio del file: {e}")


if __name__ == "__main__":
    main()