"""
Questo script orchestra un processo di acquisizione dati in tre fasi:
1.  Recupera un elenco di URI (Uniform Resource Identifier) di entità di tipo dbo:Film
    interrogando l'endpoint SPARQL pubblico di DBpedia.
2.  Per ciascun URI, esegue una seconda interrogazione SPARQL per estrarre un insieme
    predefinito di triple (dati strutturati) e l'abstract testuale in lingua inglese.
3.  Applica una logica di pulizia e formattazione, che include la compressione degli URI
    in notazione prefissata (QName) e il salvataggio dei dati curati in un file JSON.

L'obiettivo è la creazione di un dataset strutturato, pulito e leggibile, pronto
per essere utilizzato in applicazioni di analisi dati o di Natural Language Processing.
"""

import json
from SPARQLWrapper import SPARQLWrapper, JSON
import time

# Definizione di un mapping di prefissi standard del Semantic Web.
# Questa best practice consente di abbreviare gli URI (Uniform Resource Identifier) completi,
# migliorando la leggibilità e la compattezza dei dati RDF (Resource Description Framework).
PREFIX_MAP = {
    "http://dbpedia.org/resource/": "dbr:",
    "http://dbpedia.org/ontology/": "dbo:",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#": "rdf:",
    "http://www.w3.org/2000/01/rdf-schema#": "rdfs:"
}


def shorten_uri(uri):
    """
    Converte un URI completo nella sua forma compatta (QName) utilizzando la mappa PREFIX_MAP.

    Questa funzione è essenziale per la normalizzazione e la leggibilità dei dati.
    Se un URI non corrisponde a nessun prefisso definito, viene restituito inalterato,
    gestendo così correttamente sia le risorse RDF sia i valori letterali (es. date, numeri).

    :param uri: La stringa dell'URI completo da accorciare.
    :return: La rappresentazione prefissata dell'URI o la stringa originale.
    """
    if not isinstance(uri, str):
        return uri
    for base, prefix in PREFIX_MAP.items():
        if uri.startswith(base):
            return uri.replace(base, prefix, 1)
    return uri


def get_film_uris(endpoint_url, limit=100):
    """
    Esegue una query SPARQL per ottenere un elenco di URI di film da DBpedia.

    La query seleziona le risorse che sono istanze della classe dbo:Film,
    utilizzando il predicato rdf:type (abbreviato con 'a' in SPARQL).
    L'uso di 'DISTINCT' garantisce l'unicità degli URI restituiti.

    :param endpoint_url: L'URL dell'endpoint SPARQL da interrogare.
    :param limit: Il numero massimo di URI da recuperare, per limitare il carico.
    :return: Una lista di stringhe, ciascuna rappresentante un URI di un film.
    """
    sparql = SPARQLWrapper(endpoint_url)

    # Query per selezionare 'limit' istanze uniche della classe dbo:Film.
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
    Estrae, filtra e struttura i dati per un singolo URI di un film.

    La funzione esegue una query SPARQL complessa che recupera:
    1. Un insieme di triple selezionate tramite una whitelist di predicati (es. dbo:director).
       Questa è una fase cruciale di data curation per garantire la rilevanza dei dati.
    2. L'abstract del film, utilizzando un blocco OPTIONAL per non scartare film che ne sono privi
       e filtrando per la lingua inglese (lang='en') per garantire la coerenza del dataset.

    Successivamente, applica una logica di pulizia che scarta i film privi di abstract o di triple
    pertinenti, garantendo un'alta qualità del dato finale.

    :param endpoint_url: L'URL dell'endpoint SPARQL.
    :param film_uri: L'URI del film da processare.
    :return: Un dizionario contenente i dati curati del film, o None se il film
             non soddisfa i criteri di qualità.
    """
    sparql = SPARQLWrapper(endpoint_url)

    # Query SPARQL per estrarre le proprietà desiderate e l'abstract in inglese.
    query = """
        PREFIX dbo: <http://dbpedia.org/ontology/>
        SELECT ?p ?o ?abstract
        WHERE {
            # Seleziona tutte le triple uscenti dall'URI del film specificato.
            <%s> ?p ?o .
            
            # Filtro "whitelist": include solo i predicati di interesse.
            FILTER(?p IN (
                dbo:director, dbo:writer, dbo:starring, dbo:producer,
                dbo:musicComposer, dbo:country, dbo:language, dbo:releaseDate,
                dbo:distributor, dbo:cinematography, dbo:editing, dbo:imdbId
            ))
            
            # Blocco opzionale per recuperare l'abstract, se esiste.
            OPTIONAL {
                <%s> dbo:abstract ?abstract .
                # Filtro sul tag di lingua per garantire la coerenza del corpus.
                FILTER (lang(?abstract) = 'en')
            }
        }
    """ % (film_uri, film_uri)

    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    try:
        results = sparql.query().convert()["results"]["bindings"]
        # Se la query non produce risultati (es. nessuna delle proprietà in whitelist), scarta l'entità.
        if not results:
            return None

        # Estrazione euristica del titolo dalla parte finale dell'URI.
        page_title = film_uri.split('/')[-1].replace('_', ' ')

        # Estrazione e pulizia dell'abstract: si isola solo il primo paragrafo.
        abstract = ""
        if "abstract" in results[0] and results[0]["abstract"]:
            full_abstract = results[0]["abstract"]["value"]
            abstract = full_abstract.split('\n')[0].strip()

        # Criterio di curatela: scarta il film se l'abstract in inglese non è presente.
        if not abstract:
            return None

        triples = []
        # Utilizzo di un set per garantire l'unicità delle triple ed evitare duplicati.
        unique_triples = set()
        for res in results:
            if 'p' not in res or 'o' not in res:
                continue

            # Applicazione della compressione URI a soggetto, predicato e oggetto della tripla.
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

        # Criterio di curatela: scarta il film se, dopo i filtri, non ha triple associate.
        if not triples:
            return None

        # Costruzione dell'oggetto JSON finale per il film.
        return {
            "title": page_title,
            "subject_uri": shorten_uri(film_uri),
            "abstract": abstract,
            "triples": triples
        }

    except Exception as e:
        # Gestione degli errori a livello di singola richiesta per non interrompere il processo globale.
        print(f"Errore nel processare {film_uri}: {e}")
        return None


def main():
    """
    Funzione principale che orchestra l'intero processo di estrazione e salvataggio.

    Definisce i parametri, invoca le funzioni di recupero dati in sequenza
    e gestisce il salvataggio finale del dataset su file.
    """
    # Definizione dei parametri di configurazione.
    DBPEDIA_SPARQL_ENDPOINT = "https://dbpedia.org/sparql"
    FILM_LIMIT = 200
    OUTPUT_FILE = "film_dataset.json"

    # Fase 1: Ottenimento della lista di URI dei film.
    film_uris = get_film_uris(DBPEDIA_SPARQL_ENDPOINT, FILM_LIMIT)
    if not film_uris:
        print("Nessun film da processare. Uscita.")
        return

    # Fase 2: Iterazione e processamento di ciascun film.
    curated_data = []
    for i, uri in enumerate(film_uris):
        print(f"Processando film {i + 1}/{len(film_uris)}: {uri}")

        # Pausa di cortesia (politeness policy) tra le richieste.
        # Questa pratica è fondamentale per non sovraccaricare l'endpoint pubblico
        # e per evitare il blocco dell'IP a causa di un eccessivo numero di richieste
        # in un breve lasso di tempo (rate limiting).
        time.sleep(0.5)

        data = get_film_data_final_cleaned(DBPEDIA_SPARQL_ENDPOINT, uri)

        # Aggiunge i dati al dataset solo se il processo di curatela ha avuto successo.
        if data:
            curated_data.append(data)

    # Fase 3: Salvataggio dei dati curati in un file JSON.
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            # ensure_ascii=False per la corretta serializzazione di caratteri non-ASCII (es. accenti).
            # indent=4 per una formattazione human-readable del file di output.
            json.dump(curated_data, f, ensure_ascii=False, indent=4)
        print(f"\nPROCESSO COMPLETATO.")
        print(f"Dati salvati con successo in '{OUTPUT_FILE}'.")
        print(f"Sono stati salvati i dati di {len(curated_data)} film su {len(film_uris)} processati.")
    except Exception as e:
        print(f"Errore durante il salvataggio del file: {e}")


# Entry point standard per l'esecuzione dello script.
if __name__ == "__main__":
    main()