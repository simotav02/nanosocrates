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
EXTRA_ID_TOKENS = [f"<extra_id_{i}>" for i in range(150)]

SPECIAL_TOKENS = [
                     PAD_TOKEN, UNK_TOKEN, SOT_TOKEN, EOT_TOKEN, SUBJ_TOKEN, PRED_TOKEN, OBJ_TOKEN,
                     TEXT_TO_RDF_TOKEN, RDF_TO_TEXT_TOKEN, CONTINUE_RDF_TOKEN, MLM_TOKEN, MASK_TOKEN
                 ] + EXTRA_ID_TOKENS

print("1/5 - Inizializzazione di un nuovo tokenizer BPE...")
tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))

print("2/5 - Configurazione del pre-tokenizer e del decoder...")

# Logica di split unificata
special_tokens_list = [re.escape(token) for token in SPECIAL_TOKENS]
prefixes = [r'dbr:', r'dbo:', r'rdf:', r'rdfs:']
unified_split_pattern = '|'.join(special_tokens_list + prefixes + [r'_', r':'])

tokenizer.pre_tokenizer = Sequence([
    Split(pattern=unified_split_pattern, behavior="isolated"),
    ByteLevel(add_prefix_space=False)
])
tokenizer.decoder = ByteLevelDecoder()

print("3.1/5 - Preparazione del trainer BPE...")
all_special_tokens_for_trainer = SPECIAL_TOKENS + prefixes + ['_', ':']
trainer = BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=all_special_tokens_for_trainer)


def corpus_iterator():
    files_to_check = [
        SOURCE_FILE,
        TARGET_FILE,
        "../dataset_pretrain/pretrain_corpus_data/pretrain_corpus.txt"
    ]
    files = [f for f in files_to_check if os.path.exists(f)]
    if not files:
        raise FileNotFoundError("Nessun file di corpus trovato per l'addestramento del tokenizer.")

    for filepath in files:
        with open(filepath, 'r', encoding='utf-8') as f:
            num_lines = sum(1 for _ in f)
        print(f"Lettura da '{os.path.basename(filepath)}' ({num_lines} righe)...")
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in tqdm(f, total=num_lines, desc=f"Processing {os.path.basename(filepath)}"):
                yield line.strip()


print("3.2/5 - Addestramento del tokenizer dal corpus...")
tokenizer.train_from_iterator(corpus_iterator(), trainer=trainer)
print("Addestramento BPE completato.")

print(f"4/5 - Salvataggio del tokenizer in '{TOKENIZER_FILE}'...")
tokenizer.save(TOKENIZER_FILE)
print("Tokenizer salvato con successo.")

print("\n" + "=" * 50)
print("5/5 - Esempio di utilizzo del tokenizer addestrato")
loaded_tokenizer = Tokenizer.from_file(TOKENIZER_FILE)

texts_to_test = [
    "<SOT> <SUBJ> dbr:A_Quiet_Little_Wedding <PRED> dbo:starring <OBJ> dbr:Roscoe_Arbuckle <EOT>",
]

print("\n--- Test di Validazione Chiave ---")
text = texts_to_test[0]
print(f"Testo Originale:      {text}")
encoded = loaded_tokenizer.encode(text)
print(f"Token (da libreria):  {encoded.tokens}")

# Controlli essenziali
assert 'dbr:' in encoded.tokens, "FAIL: 'dbr:' non è un token singolo"
assert 'dbo:' in encoded.tokens, "FAIL: 'dbo:' non è un token singolo"
assert '_' in encoded.tokens, "FAIL: '_' non è un token singolo"
print("✅ Check strutturali RDF superati. Il tokenizer è pronto.")

decoded_text = loaded_tokenizer.decode(encoded.ids, skip_special_tokens=False)
print(f"Testo decodificato:   {decoded_text}")
assert decoded_text == text, "ERRORE: La decodifica non corrisponde all'originale!"
print("✅ Test di reversibilità superato.")

print("\n" + "=" * 50)
print("PROCESSO DEL TOKENIZER COMPLETATO CON SUCCESSO.")