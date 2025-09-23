# analyze_dataset.py

from collections import defaultdict
import os

# Assicurati che il percorso al file di corpus sia corretto
# Questo percorso assume che lo script sia nella directory principale del progetto
# e che il dataset sia in una cartella 'dataset'
CORPUS_FILE = "../dataset/training_corpus_1000.txt"

# Controlla se il file esiste prima di provare ad aprirlo
if not os.path.exists(CORPUS_FILE):
    print(f"Errore: Il file '{CORPUS_FILE}' non è stato trovato.")
    print(
        "Assicurati che il percorso sia corretto e che lo script venga eseguito dalla directory principale del tuo progetto.")
else:
    task_counts = defaultdict(int)
    total_lines = 0

    with open(CORPUS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            # Salta le righe vuote
            if not line.strip():
                continue

            total_lines += 1
            try:
                src, _ = line.split('\t', 1)
                if "<Text2RDF>" in src:
                    task_counts["Text2RDF"] += 1
                elif "<RDF2Text>" in src:
                    task_counts["RDF2Text"] += 1
                elif "<MASK>" in src:
                    task_counts["MLM"] += 1
                elif "<CONTINUERDF>" in src:
                    task_counts["ContinueRDF"] += 1
                else:
                    task_counts["Unknown"] += 1
            except ValueError:
                task_counts["Malformed (no tab)"] += 1

    print("--- Analisi del Bilanciamento dei Task nel Dataset ---")
    print(f"Totale righe analizzate: {total_lines}\n")

    for task, count in task_counts.items():
        percentage = (count / total_lines) * 100 if total_lines > 0 else 0
        print(f"- Task: {task:<15} | Conteggio: {count:<10} | Percentuale: {percentage:.2f}%")