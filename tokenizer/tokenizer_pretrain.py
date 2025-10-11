# tokenizer_pretrain.py (Versione Finale con Stampe Chiarificatrici)

import re
import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Sequence, Split, ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tqdm import tqdm

VOCAB_SIZE = 32000
CORPUS_FILE = "../dataset_pretrain/pretrain_corpus_data/pretrain_corpus.txt"
TOKENIZER_FILE = "film_corpus_bpe_tokenizer_t5.json"

# Definiamo TUTTI i token speciali che useremo nel progetto
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

ALL_SPECIAL_TOKENS = [
    PAD_TOKEN, UNK_TOKEN, SOT_TOKEN, EOT_TOKEN, SUBJ_TOKEN, PRED_TOKEN, OBJ_TOKEN,
    TEXT_TO_RDF_TOKEN, RDF_TO_TEXT_TOKEN, CONTINUE_RDF_TOKEN, MLM_TOKEN, MASK_TOKEN
] + EXTRA_ID_TOKENS

print("1/5 - Inizializzazione di un nuovo tokenizer BPE...")
tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))

print("2/5 - Configurazione del pre-tokenizer...")
special_tokens_pattern = '|'.join([re.escape(token) for token in ALL_SPECIAL_TOKENS])
prefixes = [r'dbr:', r'dbo:', r'rdf:', r'rdfs:']
rdf_chars_pattern = r'[_:]'
unified_split_pattern = '|'.join([special_tokens_pattern] + prefixes + [rdf_chars_pattern])

tokenizer.pre_tokenizer = Sequence([
    Split(pattern=unified_split_pattern, behavior="isolated"),
    ByteLevel(add_prefix_space=False)
])
tokenizer.decoder = ByteLevelDecoder()

print("3/5 - Preparazione del trainer BPE...")
# Il trainer BPE deve conoscere i token strutturali di base per non spezzarli
trainer_special_tokens = [
    PAD_TOKEN, UNK_TOKEN, SOT_TOKEN, EOT_TOKEN, SUBJ_TOKEN, PRED_TOKEN, OBJ_TOKEN,
    'dbr:', 'dbo:', 'rdf:', 'rdfs:', '_', ':'
]
trainer = BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=trainer_special_tokens)

def corpus_iterator():
    if not os.path.exists(CORPUS_FILE):
        raise FileNotFoundError(f"File del corpus puro non trovato in '{CORPUS_FILE}'. Esegui create_pretrain_corpus.py.")
    print(f"Lettura dal corpus puro '{os.path.basename(CORPUS_FILE)}'...")
    with open(CORPUS_FILE, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Processing corpus"):
            yield line.strip()

print("4/5 - Addestramento del tokenizer dal corpus PURO...")
tokenizer.train_from_iterator(corpus_iterator(), trainer=trainer)
print("Addestramento BPE completato.")
# --- NUOVA STAMPA CHIARIFICATRICE (la tua osservazione) ---
vocab_size_before_add = tokenizer.get_vocab_size()
print(f"Dimensione del vocabolario dopo BPE training: {vocab_size_before_add} (target era {VOCAB_SIZE})")


# Ora aggiungiamo TUTTI i token speciali. Molti sono già presenti dai `special_tokens` del trainer.
# Questa chiamata aggiunge solo quelli mancanti (principalmente i token di task e le maschere).
print(f"Aggiunta di {len(ALL_SPECIAL_TOKENS)} token speciali al vocabolario finale...")
tokenizer.add_special_tokens(ALL_SPECIAL_TOKENS)
vocab_size_after_add = tokenizer.get_vocab_size()
print(f"Dimensione finale del vocabolario: {vocab_size_after_add}. Aggiunti {vocab_size_after_add - vocab_size_before_add} nuovi token speciali.")

print(f"5/5 - Salvataggio del tokenizer in '{TOKENIZER_FILE}'...")
tokenizer.save(TOKENIZER_FILE)
print("Tokenizer salvato con successo.")

print("\n--- Verifica del Tokenizer Finale ---")
loaded_tokenizer = Tokenizer.from_file(TOKENIZER_FILE)
print(f"Verifica dimensione vocabolario caricato: {loaded_tokenizer.get_vocab_size()}")
assert loaded_tokenizer.token_to_id('<extra_id_99>') is not None, "FAIL: I token <extra_id_X> non sono nel vocabolario!"
assert loaded_tokenizer.token_to_id('<Text2RDF>') is not None, "FAIL: I token di task non sono nel vocabolario!"
print("✅ Tutti i token speciali sono presenti e correttamente registrati.")