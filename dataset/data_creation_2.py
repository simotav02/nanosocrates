import json
from SPARQLWrapper import SPARQLWrapper, JSON
from collections import defaultdict
import time
from tqdm import tqdm

DBPEDIA_SPARQL_ENDPOINT = "https://dbpedia.org/sparql"
FILM_LIMIT = 5000
PAGE_SIZE = 500
OUTPUT_FILE = f"film_dataset_{FILM_LIMIT}_cleaned.json"

PREFIX_MAP = {
    "http://dbpedia.org/resource/": "dbr:",
    "http://dbpedia.org/ontology/": "dbo:",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#": "rdf:",
    "http://www.w3.org/2000/01/rdf-schema#": "rdfs:"
}

WHITELISTED_PREDICATES = [
    "dbo:director", "dbo:writer", "dbo:starring", "dbo:producer",
    "dbo:musicComposer", "dbo:country", "dbo:language", "dbo:releaseDate",
    "dbo:distributor", "dbo:cinematography", "dbo:editing", "dbo:imdbId",
    "rdf:type", "rdfs:label"
]


def shorten_uri(uri):
    if not isinstance(uri, str):
        return uri
    for base, prefix in PREFIX_MAP.items():
        if uri.startswith(base):
            return uri.replace(base, prefix, 1)
    return uri


def fetch_and_process_film_data(endpoint_url, total_limit, page_size):
    sparql = SPARQLWrapper(endpoint_url)
    predicate_filter_string = ", ".join(WHITELISTED_PREDICATES)

    films_data = defaultdict(lambda: {"triples": set(), "abstract": None, "title": None})
    num_pages = (total_limit + page_size - 1) // page_size

    for i in range(num_pages):
        offset = i * page_size
        print(f"--- Esecuzione query pagina {i + 1}/{num_pages} (OFFSET {offset}, LIMIT {page_size}) ---")


        query = f"""
            PREFIX dbo: <http://dbpedia.org/ontology/>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

            SELECT ?film ?filmLabel ?abstract ?p ?o
            WHERE {{
                # Subquery per ottenere una pagina di film in modo stabile
                {{
                    SELECT DISTINCT ?film WHERE {{ ?film a dbo:Film . }} ORDER BY ?film LIMIT {page_size} OFFSET {offset}
                }}

                ?film ?p ?o .

                # --- FILTRO LINGUA POTENZIATO ---
                # Mantiene gli URI (che non sono letterali) OPPURE i letterali in inglese OPPURE i letterali senza lingua
                FILTER(!isLiteral(?o) || lang(?o) = 'en' || lang(?o) = "")

                # Filtra solo i predicati che ci interessano
                FILTER(?p IN ({predicate_filter_string}))

                # Recupera titolo e abstract in inglese (opzionali)
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
            print(f"Errore durante la query SPARQL (offset {offset}): {e}")
            print("Attendo 10 secondi e riprovo...")
            time.sleep(10)
            continue

        for res in tqdm(results, desc=f"Processando risultati pagina {i + 1}"):
            film_uri = res['film']['value']

            if not films_data[film_uri]['title'] and 'filmLabel' in res:
                films_data[film_uri]['title'] = res['filmLabel']['value']

            if not films_data[film_uri]['abstract'] and 'abstract' in res:
                films_data[film_uri]['abstract'] = res['abstract']['value'].split('\n')[0].strip()

            predicate_val = res['p']['value']
            object_val = res['o']['value']

            if predicate_val == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type":
                if not object_val.startswith("http://dbpedia.org/ontology/"):
                    continue

            films_data[film_uri]['triples'].add((
                shorten_uri(film_uri),
                shorten_uri(predicate_val),
                shorten_uri(object_val)
            ))

        time.sleep(1)

    print("\n--- Fase di pulizia e formattazione finale ---")
    curated_data = []
    for uri, data in tqdm(films_data.items(), desc="Finalizzando i record"):
        # Scarta i film senza abstract o senza triple utili
        if not data['abstract'] or not data['triples']:
            continue

        title = data['title'] if data['title'] else uri.split('/')[-1].replace('_', ' ')

        curated_data.append({
            "title": title,
            "subject_uri": shorten_uri(uri),
            "abstract": data['abstract'],
            "triples": [{"subject": s, "predicate": p, "object": o} for s, p, o in sorted(list(data['triples']))]
        })

    return curated_data


def main():
    print(f"--- Inizio acquisizione dati per {FILM_LIMIT} film ---")
    final_data = fetch_and_process_film_data(DBPEDIA_SPARQL_ENDPOINT, total_limit=FILM_LIMIT, page_size=PAGE_SIZE)

    if not final_data:
        print("ERRORE: Nessun dato recuperato. Controlla la connessione o l'endpoint SPARQL. Uscita.")
        return

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