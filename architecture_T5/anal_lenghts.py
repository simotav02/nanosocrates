# analyze_lengths.py

import os
from tokenizers import Tokenizer
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURAZIONE ---
# Assicurati che questi percorsi siano corretti rispetto alla posizione dello script.
# Stiamo analizzando i dati del "quick test".
TOKENIZER_PATH = "../tokenizer/film_corpus_bpe_tokenizer.json"
DATA_DIR = "../dataset/training_data"  # La cartella con i dati per il test rapido

# Il valore di seq_len che vogliamo testare
SEQ_LEN_TO_TEST = 512


# --- SCRIPT DI ANALISI ---

def analyze_token_lengths(tokenizer_path, data_dir):
    """
    Carica il tokenizer e i dati, calcola le lunghezze delle sequenze tokenizzate
    e stampa statistiche dettagliate.
    """
    print(f"1/4 - Caricamento del tokenizer da: {tokenizer_path}")
    if not os.path.exists(tokenizer_path):
        print(f"ERRORE: Tokenizer non trovato al percorso: {tokenizer_path}")
        return
    tokenizer = Tokenizer.from_file(tokenizer_path)

    source_file = os.path.join(data_dir, "train.source")
    target_file = os.path.join(data_dir, "train.target")

    if not os.path.exists(source_file) or not os.path.exists(target_file):
        print(f"ERRORE: File di dati non trovati nella cartella: {data_dir}")
        return

    print("2/4 - Analisi delle lunghezze dei token (potrebbe richiedere un minuto)...")
    source_lengths = []
    target_lengths = []

    with open(source_file, 'r', encoding='utf-8') as f_src, \
            open(target_file, 'r', encoding='utf-8') as f_tgt:

        lines = list(zip(f_src, f_tgt))
        for src_line, tgt_line in tqdm(lines, desc="Processando le righe"):
            # Tokenizza e calcola la lunghezza per source e target
            src_tokens = tokenizer.encode(src_line.strip())
            tgt_tokens = tokenizer.encode(tgt_line.strip())
            source_lengths.append(len(src_tokens.ids))
            target_lengths.append(len(tgt_tokens.ids))

    if not source_lengths or not target_lengths:
        print("Nessun dato da analizzare.")
        return

    print("\n3/4 - Calcolo delle statistiche...")

    # --- Statistiche per le sequenze SORGENTE ---
    print("\n--- Statistiche Lunghezze SORGENTE (Source) ---")
    max_len_src = np.max(source_lengths)
    avg_len_src = np.mean(source_lengths)
    median_len_src = np.median(source_lengths)
    p95_src = np.percentile(source_lengths, 95)
    p98_src = np.percentile(source_lengths, 98)
    troncate_src = sum(1 for length in source_lengths if length > SEQ_LEN_TO_TEST)

    print(f"Lunghezza Massima: {max_len_src}")
    print(f"Lunghezza Media: {avg_len_src:.2f}")
    print(f"Lunghezza Mediana: {median_len_src}")
    print(f"95° Percentile: {p95_src:.2f} (il 95% delle sequenze è più corto di questo valore)")
    print(f"98° Percentile: {p98_src:.2f} (il 98% delle sequenze è più corto di questo valore)")
    print(
        f"Con seq_len={SEQ_LEN_TO_TEST}, verrebbero troncate {troncate_src} sequenze su {len(source_lengths)} ({troncate_src / len(source_lengths) * 100:.2f}%)")

    # --- Statistiche per le sequenze TARGET ---
    print("\n--- Statistiche Lunghezze DESTINAZIONE (Target) ---")
    max_len_tgt = np.max(target_lengths)
    avg_len_tgt = np.mean(target_lengths)
    median_len_tgt = np.median(target_lengths)
    p95_tgt = np.percentile(target_lengths, 95)
    p98_tgt = np.percentile(target_lengths, 98)
    troncate_tgt = sum(1 for length in target_lengths if length > SEQ_LEN_TO_TEST)

    print(f"Lunghezza Massima: {max_len_tgt}")
    print(f"Lunghezza Media: {avg_len_tgt:.2f}")
    print(f"Lunghezza Mediana: {median_len_tgt}")
    print(f"95° Percentile: {p95_tgt:.2f}")
    print(f"98° Percentile: {p98_tgt:.2f}")
    print(
        f"Con seq_len={SEQ_LEN_TO_TEST}, verrebbero troncate {troncate_tgt} sequenze su {len(target_lengths)} ({troncate_tgt / len(target_lengths) * 100:.2f}%)")

    # --- Visualizzazione ---
    print("\n4/4 - Generazione del grafico della distribuzione delle lunghezze...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.hist(source_lengths, bins=50, color='skyblue', edgecolor='black')
    ax1.axvline(SEQ_LEN_TO_TEST, color='r', linestyle='--', linewidth=2, label=f'seq_len = {SEQ_LEN_TO_TEST}')
    ax1.set_title('Distribuzione Lunghezze SORGENTE')
    ax1.set_xlabel('Lunghezza della sequenza (numero di token)')
    ax1.set_ylabel('Frequenza')
    ax1.legend()

    ax2.hist(target_lengths, bins=50, color='salmon', edgecolor='black')
    ax2.axvline(SEQ_LEN_TO_TEST, color='r', linestyle='--', linewidth=2, label=f'seq_len = {SEQ_LEN_TO_TEST}')
    ax2.set_title('Distribuzione Lunghezze TARGET')
    ax2.set_xlabel('Lunghezza della sequenza (numero di token)')
    ax2.set_ylabel('Frequenza')
    ax2.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Assicurati di aver installato numpy e matplotlib:
    # pip install numpy matplotlib
    analyze_token_lengths(TOKENIZER_PATH, DATA_DIR)