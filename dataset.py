import json
from SPARQLWrapper import SPARQLWrapper, JSON
import time

# Definiamo i prefissi comuni una sola volta per riutilizzarli.
PREFIX_MAP = {
    "http://dbpedia.org/resource/": "dbr:",
    "http://dbpedia.org/ontology/": "dbo:",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#": "rdf:",
    "http://www.w3.org/2000/01/rdf-schema#": "rdfs:"
}


def shorten_uri(uri):
    """Sostituisce un URI completo con la sua versione prefissata, se applicabile."""
    if not isinstance(uri, str):
        return uri
    # Itera sulla mappa e sostituisce la base dell'URI con il prefisso corrispondente.
    for base, prefix in PREFIX_MAP.items():
        if uri.startswith(base):
            # Usiamo replace con count=1 per essere sicuri di sostituire solo l'inizio.
            return uri.replace(base, prefix, 1)
    # Se nessun prefisso corrisponde, restituisce la stringa originale (es. un letterale come una data o un ID).
    return uri


def get_film_uris(endpoint_url, limit=100):
    """Recupera una lista di URI di film da DBpedia."""
    sparql = SPARQLWrapper(endpoint_url)
    query = """
        PREFIX dbo: <http://dbpedia.org/ontology/>
        SELECT DISTINCT ?film
        WHERE {
          ?film a dbo:Film .
        }
        LIMIT %d
    """ % limit
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    film_uris = []
    try:
        results = sparql.query().convert()
        for result in results["results"]["bindings"]:
            film_uris.append(result["film"]["value"])
        print(f"Trovati {len(film_uris)} URI di film.")
    except Exception as e:
        print(f"Errore durante la query per ottenere i film: {e}")
    return film_uris


def get_film_data_final_cleaned(endpoint_url, film_uri):
    """
    Recupera l'abstract (primo paragrafo) e le triple filtrate (con whitelist),
    e comprime tutti gli URI usando i prefissi standard.
    """
    sparql = SPARQLWrapper(endpoint_url)

    query = """
        PREFIX dbo: <http://dbpedia.org/ontology/>
        SELECT ?p ?o ?abstract
        WHERE {
            <%s> ?p ?o .
            FILTER(?p IN (
                dbo:director, dbo:writer, dbo:starring, dbo:producer,
                dbo:musicComposer, dbo:country, dbo:language, dbo:releaseDate,
                dbo:distributor, dbo:cinematography, dbo:editing, dbo:imdbId
            ))
            OPTIONAL {
                <%s> dbo:abstract ?abstract .
                FILTER (lang(?abstract) = 'en')
            }
        }
    """ % (film_uri, film_uri)

    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    try:
        results = sparql.query().convert()["results"]["bindings"]
        if not results:
            return None

        page_title = film_uri.split('/')[-1].replace('_', ' ')

        abstract = ""
        if "abstract" in results[0] and results[0]["abstract"]:
            full_abstract = results[0]["abstract"]["value"]
            abstract = full_abstract.split('\n')[0].strip()

        if not abstract:
            return None

        triples = []
        unique_triples = set()
        for res in results:
            if 'p' not in res or 'o' not in res:
                continue

            subject_short = shorten_uri(film_uri)
            predicate_short = shorten_uri(res['p']['value'])
            object_short = shorten_uri(res['o']['value'])

            triple_tuple = (subject_short, predicate_short, object_short)

            if triple_tuple not in unique_triples:
                triples.append({
                    "subject": subject_short,
                    "predicate": predicate_short,
                    "object": object_short
                })
                unique_triples.add(triple_tuple)

        if not triples:
            return None

        return {
            "title": page_title,
            "subject_uri": shorten_uri(film_uri),
            "abstract": abstract,
            "triples": triples
        }

    except Exception as e:
        print(f"Errore nel processare {film_uri}: {e}")
        return None


def main():
    """Funzione principale per orchestrare la raccolta e il salvataggio dei dati."""
    DBPEDIA_SPARQL_ENDPOINT = "https://dbpedia.org/sparql"
    FILM_LIMIT = 200
    OUTPUT_FILE = "film_dataset.json"

    film_uris = get_film_uris(DBPEDIA_SPARQL_ENDPOINT, FILM_LIMIT)
    if not film_uris:
        print("Nessun film da processare. Uscita.")
        return

    curated_data = []
    for i, uri in enumerate(film_uris):
        print(f"Processando film {i + 1}/{len(film_uris)}: {uri}")
        #time.sleep(0.5)
        data = get_film_data_final_cleaned(DBPEDIA_SPARQL_ENDPOINT, uri)
        if data:
            curated_data.append(data)

    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(curated_data, f, ensure_ascii=False, indent=4)
        print(f"\nPROCESSO COMPLETATO.")
        print(f"Dati salvati con successo in '{OUTPUT_FILE}'.")
        print(f"Sono stati salvati i dati di {len(curated_data)} film su {len(film_uris)} processati.")
    except Exception as e:
        print(f"Errore durante il salvataggio del file: {e}")


if __name__ == "__main__":
    main()