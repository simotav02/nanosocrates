import re
import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Sequence, Split, ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tqdm import tqdm

VOCAB_SIZE = 32000
TRAINING_DATA_DIR = "../dataset/training_data_cleaned"
SOURCE_FILE = os.path.join(TRAINING_DATA_DIR, "train.source")
TARGET_FILE = os.path.join(TRAINING_DATA_DIR, "train.target")
TOKENIZER_FILE = "film_corpus_bpe_tokenizer_t5.json"

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
SOT_TOKEN = "<SOT>"
EOT_TOKEN = "<EOT>"
SUBJ_TOKEN = "<SUBJ>"
PRED_TOKEN = "<PRED>"
OBJ_TOKEN = "<OBJ>"
MASK_TOKEN = "<MASK>"
TEXT_TO_RDF_TOKEN = "<Text2RDF>"
RDF_TO_TEXT_TOKEN = "<RDF2Text>"
CONTINUE_RDF_TOKEN = "<CONTINUERDF>"
MLM_TOKEN = "<MLM>"
EXTRA_ID_TOKENS = [f"<extra_id_{i}>" for i in range(100)]

SPECIAL_TOKENS = [
    PAD_TOKEN, UNK_TOKEN, SOT_TOKEN, EOT_TOKEN, SUBJ_TOKEN, PRED_TOKEN, OBJ_TOKEN,
    TEXT_TO_RDF_TOKEN, RDF_TO_TEXT_TOKEN, CONTINUE_RDF_TOKEN, MLM_TOKEN, MASK_TOKEN
] + EXTRA_ID_TOKENS


print("1/5 - Inizializzazione di un nuovo tokenizer BPE...")
tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))

print("2/5 - Configurazione del pre-tokenizer e del decoder...")


special_tokens_pattern = f"({'|'.join(re.escape(token) for token in SPECIAL_TOKENS)})"

tokenizer.pre_tokenizer = Sequence([
    Split(pattern=special_tokens_pattern, behavior="isolated"),
    ByteLevel(add_prefix_space=False)
])
tokenizer.decoder = ByteLevelDecoder()

print("3/5 - Preparazione del trainer BPE...")
trainer = BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=SPECIAL_TOKENS)

def corpus_iterator():
    files = [SOURCE_FILE, TARGET_FILE]
    for filepath in files:
        if not os.path.exists(filepath):
            print(f"ATTENZIONE: File del corpus non trovato: {filepath}. Verrà saltato.")
            continue
        with open(filepath, 'r', encoding='utf-8') as f:
            num_lines = sum(1 for _ in f)
        print(f"Lettura da '{filepath}' ({num_lines} righe)...")
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in tqdm(f, total=num_lines, desc=f"Processing {os.path.basename(filepath)}"):
                yield line.strip()

print("4/5 - Addestramento del tokenizer dal corpus...")
tokenizer.train_from_iterator(corpus_iterator(), trainer=trainer)
print("Addestramento BPE completato.")

print(f"5/5 - Salvataggio del tokenizer in '{TOKENIZER_FILE}'...")
tokenizer.save(TOKENIZER_FILE)
print("Tokenizer salvato con successo.")

print("\n" + "="*50)
print("--- Esempio di utilizzo del tokenizer addestrato ---")
loaded_tokenizer = Tokenizer.from_file(TOKENIZER_FILE)

texts_to_test = [
    f"Inception is a 2010 science fiction action film written and directed by Christopher Nolan. {TEXT_TO_RDF_TOKEN}",
    f"<SOT> <SUBJ> dbr:Inception <PRED> dbo:director <OBJ> dbr:Christopher_Nolan <EOT> {RDF_TO_TEXT_TOKEN}",
    f"The film <extra_id_0> was directed by <extra_id_1> Nolan.",
    f"<extra_id_0> Inception <extra_id_1> Christopher"
]

for i, text in enumerate(texts_to_test):
    print(f"\n--- Test {i + 1} ---")
    print(f"Testo Originale:      {text}")
    encoded = loaded_tokenizer.encode(text)
    print(f"Token (da libreria):  {encoded.tokens}")
    decoded_text = loaded_tokenizer.decode(encoded.ids, skip_special_tokens=False)
    print(f"Testo decodificato:   {decoded_text}")
    assert decoded_text == text, "ERRORE: La decodifica non corrisponde all'originale!"
    print("✅ Test superato: La decodifica è corretta e reversibile.")

print("\n" + "="*50)