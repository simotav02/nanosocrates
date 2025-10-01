import re
import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Sequence, Split, ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

# Definizioni dei token speciali (invariate)
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
SOT_TOKEN = "<SOT>"
EOT_TOKEN = "<EOT>"
SUBJ_TOKEN = "<SUBJ>"
PRED_TOKEN = "<PRED>"
OBJ_TOKEN = "<OBJ>"
TEXT_TO_RDF_TOKEN = "<Text2RDF>"
RDF_TO_TEXT_TOKEN = "<RDF2Text>"
CONTINUE_RDF_TOKEN = "<CONTINUERDF>"
MLM_TOKEN = "<MLM>"
MASK_TOKEN = "<MASK>"

SPECIAL_TOKENS = [
    PAD_TOKEN, UNK_TOKEN, SOT_TOKEN, EOT_TOKEN, SUBJ_TOKEN, PRED_TOKEN, OBJ_TOKEN,
    TEXT_TO_RDF_TOKEN, RDF_TO_TEXT_TOKEN, CONTINUE_RDF_TOKEN, MLM_TOKEN, MASK_TOKEN
]

VOCAB_SIZE = 32000
# Assicurati che punti alla versione più recente e pulita del tuo dataset
TRAINING_DATA_DIR = "../dataset/training_data_cleaned"
SOURCE_FILE = os.path.join(TRAINING_DATA_DIR, "train.source")
TARGET_FILE = os.path.join(TRAINING_DATA_DIR, "train.target")
# Diamo un nome finale e chiaro
TOKENIZER_FILE = "film_corpus_bpe_tokenizer.json"

print("1/4 - Inizializzazione di un nuovo tokenizer BPE...")
# Usiamo il modello BPE base, come facevi tu
tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))

print("2/4 - Configurazione del pre-tokenizer e del decoder...")
# --- RITORNO ALLA TUA LOGICA ORIGINALE E CORRETTA ---
# Questa sequenza gestisce correttamente sia i token speciali che qualsiasi altro carattere.
# Split isola i token speciali, ByteLevel gestisce tutto il resto a livello di byte,
# garantendo che non ci siano mai caratteri <UNK>.
tokenizer.pre_tokenizer = Sequence([
    Split(pattern=f"({'|'.join(re.escape(token) for token in SPECIAL_TOKENS)})", behavior="isolated"),
    ByteLevel(add_prefix_space=False)
])
tokenizer.decoder = ByteLevelDecoder()

print("3/4 - Preparazione del trainer BPE...")
trainer = BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=SPECIAL_TOKENS)

def corpus_iterator():
    files = [SOURCE_FILE, TARGET_FILE]
    for filepath in files:
        if not os.path.exists(filepath):
            print(f"ATTENZIONE: File del corpus non trovato: {filepath}. Verrà saltato.")
            continue
        print(f"Lettura dal file: {filepath}")
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                yield line.strip()

print("3/4 - Addestramento del tokenizer dal corpus...")
tokenizer.train_from_iterator(corpus_iterator(), trainer=trainer)
print("Addestramento BPE completato.")

print(f"4/4 - Salvataggio del tokenizer in '{TOKENIZER_FILE}'...")
tokenizer.save(TOKENIZER_FILE)
print("Tokenizer salvato con successo.")

# --- ESEMPIO D'USO ---
print("\n--- Esempio di utilizzo del tokenizer addestrato ---")
loaded_tokenizer = Tokenizer.from_file(TOKENIZER_FILE)

texts_to_test = [
    f"<SOT> <SUBJ> dbr:Cabaret_(1972_film) <PRED> {MASK_TOKEN} <OBJ> dbr:Ralph_Burns <EOT> <MLM>",
    f"Caddyshack is a 1980 American sports comedy film directed by Harold Ramis {TEXT_TO_RDF_TOKEN}",
    f"<SOT> <SUBJ> dbr:California_(1947_film) <PRED> dbo:director <OBJ> dbr:John_Farrow <EOT> {RDF_TO_TEXT_TOKEN}",
    f"<SOT> <SUBJ> dbr:Cain_XVIII <PRED> dbo:imdbId <OBJ> 0176875 <EOT> {CONTINUE_RDF_TOKEN}"
]

for i, text in enumerate(texts_to_test):
    print(f"\n--- Test {i + 1} ---")
    print(f"Testo Originale:      {text}")

    encoded = loaded_tokenizer.encode(text)
    print(f"Token (da libreria):  {encoded.tokens}")

    decoded_text = loaded_tokenizer.decode(encoded.ids, skip_special_tokens=False)
    print(f"Testo decodificato:   {decoded_text}")

    assert decoded_text == text, f"La decodifica non corrisponde all'originale!\nOriginale: '{text}'\nDecodificato: '{decoded_text}'"
    print("Test superato: Decodifica corretta.")