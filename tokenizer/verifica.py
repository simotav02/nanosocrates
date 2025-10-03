# verify_tokenizer.py

from tokenizers import Tokenizer
import os

TOKENIZER_PATH = "film_corpus_bpe_tokenizer_t5.json"


def main():
    print(f"--- VERIFICA DEL TOKENIZER IN '{TOKENIZER_PATH}' ---")

    if not os.path.exists(TOKENIZER_PATH):
        print(f"❌ ERRORE CRITICO: Il file del tokenizer non è stato trovato.")
        print("Assicurati di aver eseguito 'tokenizer_lib.py' con successo.")
        return

    try:
        tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    except Exception as e:
        print(f"❌ ERRORE CRITICO: Impossibile caricare il file del tokenizer. Potrebbe essere corrotto. Errore: {e}")
        return

    print("✅ Tokenizer caricato con successo.")

    vocab_size = tokenizer.get_vocab_size()
    print(f"Dimensione del vocabolario: {vocab_size}")

    if vocab_size < 150:  # Se il vocabolario è minuscolo, l'addestramento è fallito
        print(
            "❌ ATTENZIONE: La dimensione del vocabolario è troppo piccola. Probabilmente l'addestramento non ha trovato i file del corpus.")
    else:
        print("✅ La dimensione del vocabolario sembra ragionevole.")

    print("\n--- Verifica dei token speciali per Span Corruption ---")

    missing_tokens = []

    # Controlliamo i primi 10 e l'ultimo dei token <extra_id_...>
    tokens_to_check = [f"<extra_id_{i}>" for i in range(10)] + ["<extra_id_99>"]

    for token_str in tokens_to_check:
        token_id = tokenizer.token_to_id(token_str)
        if token_id is None:
            missing_tokens.append(token_str)
            print(f"❌ Trovato token MANCANTE: '{token_str}' -> ID: {token_id}")
        else:
            print(f"✅ Trovato token presente: '{token_str}' -> ID: {token_id}")

    print("\n--- Risultato della Verifica ---")
    if not missing_tokens:
        print("🎉 OTTIMO! Tutti i token <extra_id_...> sono stati trovati nel vocabolario.")
        print("Il tokenizer sembra essere stato creato correttamente.")
    else:
        print("❌ PROBLEMA RILEVATO: Il tokenizer è difettoso. Mancano i token per lo Span Corruption.")
        print(
            "Causa probabile: lo script 'tokenizer_lib.py' è stato eseguito quando i file del corpus non erano disponibili.")
        print(
            "Soluzione: 1. Cancella 'film_corpus_bpe_tokenizer_t5.json'. 2. Assicurati che i dati in 'training_data_cleaned' esistano. 3. Riesegui 'tokenizer_lib.py'.")


if __name__ == "__main__":
    main()