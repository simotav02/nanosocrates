import os
from collections import Counter

INPUT_DIR = "training_data_cleaned"

TEXT_TO_RDF_TOKEN = "<Text2RDF>"
RDF_TO_TEXT_TOKEN = "<RDF2Text>"
CONTINUE_RDF_TOKEN = "<CONTINUERDF>"
MLM_TOKEN = "<MLM>"


def analyze_training_corpus(directory):
    """
    Analizza il file di input del training e calcola la distribuzione dei task.
    """
    source_filepath = os.path.join(directory, "train.source")

    if not os.path.exists(source_filepath):
        print(f"Errore: File '{source_filepath}' non trovato.")
        print("Assicurati di aver prima eseguito lo script di generazione del corpus.")
        return

    print(f"--- Inizio Analisi del Corpus in '{directory}' ---")

    task_counts = Counter()
    total_lines = 0

    try:
        with open(source_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                total_lines += 1
                line = line.strip()

                if line.endswith(TEXT_TO_RDF_TOKEN):
                    task_counts['Text2RDF'] += 1
                elif line.endswith(RDF_TO_TEXT_TOKEN):
                    task_counts['RDF2Text'] += 1
                elif line.endswith(CONTINUE_RDF_TOKEN):
                    task_counts['CONTINUERDF'] += 1
                elif line.endswith(MLM_TOKEN):
                    task_counts['MLM'] += 1
                else:
                    task_counts['Unknown'] += 1

        print(f"Analisi completata. Trovati {total_lines} esempi totali.")
        print("-" * 40)

        print("Distribuzione dei Task nel Dataset:")
        print("-" * 40)

        tasks_in_order = ['Text2RDF', 'RDF2Text', 'CONTINUERDF', 'MLM']

        for task_name in tasks_in_order:
            count = task_counts[task_name]
            if total_lines > 0:
                percentage = (count / total_lines) * 100
                print(f"- Task '{task_name}': {count:6d} esempi ({percentage:5.2f}%)")
            else:
                print(f"- Task '{task_name}': {count:6d} esempi (N/A)")

        if task_counts['Unknown'] > 0:
            print(f"- Task 'Unknown': {task_counts['Unknown']:6d} esempi (!!!)")
            print("\nATTENZIONE: Trovati esempi con formato non riconosciuto.")

        print("-" * 40)

        generative_tasks_count = (task_counts['Text2RDF'] +
                                  task_counts['RDF2Text'] +
                                  task_counts['CONTINUERDF'])

        mlm_task_count = task_counts['MLM']

        if total_lines > 0:
            generative_percentage = (generative_tasks_count / total_lines) * 100
            mlm_percentage = (mlm_task_count / total_lines) * 100

            print("Riepilogo del Bilanciamento per Tipologia:")
            print(
                f"  - Task Generativi (Text2RDF, RDF2Text, CONTINUERDF): {generative_tasks_count} esempi ({generative_percentage:.2f}%)")
            print(
                f"  - Task di Completamento (MLM):                      {mlm_task_count} esempi ({mlm_percentage:.2f}%)")

        print("\n--- Analisi Conclusa ---")

    except Exception as e:
        print(f"Si è verificato un errore durante la lettura del file: {e}")


if __name__ == "__main__":
    analyze_training_corpus(INPUT_DIR)