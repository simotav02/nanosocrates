import json
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
# Import aggiornati
from tokenizers.pre_tokenizers import ByteLevel, Sequence, Whitespace, Split
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

# -----------------------------------------------------------------------------
# STEP 1: CONFIGURAZIONE
# -----------------------------------------------------------------------------
SPECIAL_TOKENS = [
    "<PAD>", "<UNK>", "<SOT>", "<EOT>", "<SUBJ>", "<PRED>", "<OBJ>",
    "<RDF2Text>", "<Text2RDF>", "<CONTINUERDF>", "<MASK>"
]
VOCAB_SIZE = 16000
CORPUS_FILE = "../dataset/training_corpus.txt"  # Assicurati che il percorso sia corretto
TOKENIZER_FILE = "nanosocrates_hf_tokenizer.json"

print("1/4 - Inizializzazione di un nuovo tokenizer BPE...")
tokenizer = Tokenizer(BPE(unk_token="<UNK>"))

# -----------------------------------------------------------------------------
# STEP 2: PRE-TOKENIZZAZIONE (VERSIONE BYTE-LEVEL - LA PIÙ ROBUSTA)
# -----------------------------------------------------------------------------
# Questa sequenza garantisce la perfetta reversibilità, inclusi gli spazi.
# 1. Prima isoliamo i token speciali per evitare che vengano processati da ByteLevel.
# 2. Poi, il resto del testo viene processato da ByteLevel, che gestisce gli spazi
#    e tutti gli altri caratteri in modo non ambiguo.
# Nota: L'ordine è importante. Qui è meglio applicare Split prima di Whitespace per ByteLevel.
tokenizer.pre_tokenizer = Sequence([
    Split(pattern=f"({'|'.join(SPECIAL_TOKENS)})", behavior="isolated"),
    ByteLevel(add_prefix_space=False)
])

# Il decoder DEVE corrispondere al pre-tokenizer per una decodifica corretta.
tokenizer.decoder = ByteLevelDecoder()

print("2/4 - Preparazione del trainer BPE...")

# -----------------------------------------------------------------------------
# STEP 3: ADDESTRAMENTO
# -----------------------------------------------------------------------------
trainer = BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=SPECIAL_TOKENS)


def corpus_iterator():
    with open(CORPUS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            # Addestriamo su input e output, li uniamo con uno spazio
            # perché il pre-tokenizer si aspetta una singola stringa.
            yield line.strip().replace('\t', ' ')


print(f"3/4 - Addestramento del tokenizer dal corpus '{CORPUS_FILE}'...")
tokenizer.train_from_iterator(corpus_iterator(), trainer=trainer)
print("Addestramento BPE completato.")

# -----------------------------------------------------------------------------
# STEP 4: SALVATAGGIO E TEST
# -----------------------------------------------------------------------------
print(f"4/4 - Salvataggio del tokenizer in '{TOKENIZER_FILE}'...")
tokenizer.save(TOKENIZER_FILE)
print("Tokenizer salvato con successo.")

# --- ESEMPIO D'USO ---
print("\n--- Esempio di utilizzo del tokenizer addestrato ---")
loaded_tokenizer = Tokenizer.from_file(TOKENIZER_FILE)

texts_to_test = [
    "<SOT> <SUBJ> dbr:Cabaret_(1972_film) <PRED> <MASK> <OBJ> dbr:Ralph_Burns <EOT>",
    "Caddyshack is a 1980 American sports comedy film directed by Harold Ramis <Text2RDF>",
    "<SOT> <SUBJ> dbr:California_(1947_film) <PRED> dbo:director <OBJ> dbr:John_Farrow <EOT> <RDF2Text>",
    "<SOT> <SUBJ> dbr:Cain_XVIII <PRED> dbo:imdbId <OBJ> 0176875 <EOT> <CONTINUERDF>"
]

for i, text in enumerate(texts_to_test):
    print(f"\n--- Test {i + 1} ---")
    print(f"Testo Originale:      {text}")

    encoded = loaded_tokenizer.encode(text)
    print(f"Token (da libreria):  {encoded.tokens}")
    print(f"ID (da libreria):     {encoded.ids}")

    # La decodifica ora dovrebbe ricostruire gli spazi correttamente
    decoded_text = loaded_tokenizer.decode(encoded.ids, skip_special_tokens=False)
    print(f"Testo decodificato:   {decoded_text}")

    assert decoded_text == text, "La decodifica non corrisponde all'originale!"
    print("Test superato: Decodifica corretta.")