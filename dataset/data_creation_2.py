"""
Questo script orchestra un processo di acquisizione dati in due fasi,
seguendo le best practice indicate dalla traccia:
1.  Esegue una SINGOLA, potente interrogazione SPARQL all'endpoint di DBpedia.
    Questa query recupera in un colpo solo i dati per un numero predefinito di film,
    includendo titolo, il primo paragrafo dell'abstract e SOLO le triple con
    predicati informativi pre-selezionati (whitelist).
2.  Processa i risultati per raggrupparli per film, formatta i dati
    (compressione URI in QName) e li salva in un file JSON.
"""

import json
from SPARQLWrapper import SPARQLWrapper, JSON
from collections import defaultdict

PREFIX_MAP = {
    "http://dbpedia.org/resource/": "dbr:",
    "http://dbpedia.org/ontology/": "dbo:",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#": "rdf:",
    "http://www.w3.org/2000/01/rdf-schema#": "rdfs:"
}

# La tua "whitelist" di predicati, definita qui per chiarezza.
# Questa è una pratica eccellente per la curatela del dataset.
WHITELISTED_PREDICATES = [
    "dbo:director", "dbo:writer", "dbo:starring", "dbo:producer",
    "dbo:musicComposer", "dbo:country", "dbo:language", "dbo:releaseDate",
    "dbo:distributor", "dbo:cinematography", "dbo:editing", "dbo:imdbId",
    "rdf:type", "rdfs:label"  # Aggiungiamo anche questi che sono utili
]


def shorten_uri(uri):
    """Converte un URI completo nella sua forma compatta (QName)."""
    if not isinstance(uri, str):
        return uri
    for base, prefix in PREFIX_MAP.items():
        if uri.startswith(base):
            return uri.replace(base, prefix, 1)
    return uri


def fetch_and_process_film_data(endpoint_url, limit=100):
    """
    Esegue una singola query per ottenere tutti i dati dei film e li processa.
    Questo approccio è drasticamente più efficiente e rispetta tutti gli hint.
    """
    sparql = SPARQLWrapper(endpoint_url)

    # Convertiamo la nostra lista di predicati in una stringa per la query SPARQL
    predicate_filter_string = ", ".join(WHITELISTED_PREDICATES)

    query = f"""
        PREFIX dbo: <http://dbpedia.org/ontology/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

        SELECT ?film ?filmLabel ?abstract ?p ?o
        WHERE {{
            {{
                SELECT DISTINCT ?film WHERE {{ ?film a dbo:Film . }} LIMIT {limit}
            }}

            ?film ?p ?o .

            # Filtro sui predicati per mantenere il dataset pulito e focalizzato
            FILTER(?p IN ({predicate_filter_string}))

            OPTIONAL {{ 
                ?film rdfs:label ?filmLabel .
                FILTER(LANG(?filmLabel) = 'en')
            }}

            OPTIONAL {{
                ?film dbo:abstract ?abstract .
                FILTER(LANG(?abstract) = 'en')
            }}
        }}
    """

    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    print(f"Esecuzione della query unica per {limit} film. Potrebbe richiedere un po' di tempo...")

    try:
        results = sparql.query().convert()["results"]["bindings"]
    except Exception as e:
        print(f"Errore critico durante la query SPARQL: {e}")
        return []

    print(f"Query completata. Ricevute {len(results)} righe. Inizio processamento...")

    films_data = defaultdict(lambda: {"triples": set(), "abstract": None, "title": None})

    for res in results:
        film_uri = res['film']['value']

        if not films_data[film_uri]['title'] and 'filmLabel' in res:
            films_data[film_uri]['title'] = res['filmLabel']['value']

        if not films_data[film_uri]['abstract'] and 'abstract' in res:
            films_data[film_uri]['abstract'] = res['abstract']['value'].split('\n')[0].strip()

        # Logica di filtraggio per le triple "rdf:type"
        predicate_val = res['p']['value']
        object_val = res['o']['value']

        # Se il predicato è "rdf:type", controlliamo l'oggetto della tripla.
        if predicate_val == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type":
            # Teniamo la tripla solo se l'oggetto appartiene all'ontologia di DBpedia (dbo:)
            # In questo modo scartiamo i tipi provenienti da YAGO, Schema.org, ecc.
            if not object_val.startswith("http://dbpedia.org/ontology/"):
                continue

        # Aggiungiamo la tripla (usando un set per evitare duplicati)
        subject_short = shorten_uri(film_uri)
        predicate_short = shorten_uri(predicate_val)
        object_short = shorten_uri(object_val)
        films_data[film_uri]['triples'].add((subject_short, predicate_short, object_short))

    # Fase di pulizia finale e formattazione (questa parte rimane invariata)
    curated_data = []
    for uri, data in films_data.items():
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
    DBPEDIA_SPARQL_ENDPOINT = "https://dbpedia.org/sparql"
    FILM_LIMIT = 100
    OUTPUT_FILE = f"film_dataset_{FILM_LIMIT}_final.json"

    final_data = fetch_and_process_film_data(DBPEDIA_SPARQL_ENDPOINT, FILM_LIMIT)

    if not final_data:
        print("Nessun dato recuperato. Uscita.")
        return

    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, ensure_ascii=False, indent=4)
        print(f"\nPROCESSO COMPLETATO.")
        print(f"Dati salvati con successo in '{OUTPUT_FILE}'.")
        print(f"Sono stati salvati i dati di {len(final_data)} film.")
    except Exception as e:
        print(f"Errore durante il salvataggio del file: {e}")


if __name__ == "__main__":
    main()