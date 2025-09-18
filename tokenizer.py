import re
import json
from collections import defaultdict
from tqdm import tqdm

# -----------------------------------------------------------------------------
# STEP 1: CONFIGURAZIONE E DEFINIZIONE DEI TOKEN SPECIALI
# -----------------------------------------------------------------------------

SPECIAL_TOKENS = [
    "<PAD>", "<UNK>", "<SOT>", "<EOT>", "<SUBJ>", "<PRED>", "<OBJ>",
    "<RDF2Text>", "<Text2RDF>", "<CONTINUERDF>", "<MASK>"
]

VOCAB_SIZE = 16000
CORPUS_FILE = "training_corpus.txt"
TOKENIZER_FILE = "nanosocrates_tokenizer.json"

# -----------------------------------------------------------------------------
# STEP 2: PRE-TOKENIZZAZIONE E CALCOLO DELLE FREQUENZE
# -----------------------------------------------------------------------------

print("1/5 - Lettura e preparazione del corpus...")
full_corpus = []
with open(CORPUS_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        for part in parts:
            full_corpus.append(part)

word_freqs = defaultdict(int)

# --- MODIFICA CHIAVE #1: REGEX MIGLIORATA ---
# Questa regex è ispirata a quella di GPT-2. Isola i token speciali
# e poi gestisce le parole, i numeri e la punteggiatura in modo più robusto.
# Lo spazio viene considerato parte del token successivo.
special_tokens_pattern = "|".join(re.escape(token) for token in SPECIAL_TOKENS)
pre_tokenizer_regex = re.compile(
    rf"""({special_tokens_pattern})|'s|'t|'re|'ve|'m|'ll|'d| ?\p{{L}}+| ?\p{{N}}+| ?[^\s\p{{L}}\p{{N}}]+|\s+(?!\S)|\s+""",
    re.UNICODE)

print("2/5 - Pre-tokenizzazione e calcolo delle frequenze...")
# Dobbiamo installare il modulo 'regex' che ha un supporto Unicode migliore
try:
    import regex
except ImportError:
    print("Installando il modulo 'regex'... Esegui di nuovo lo script dopo l'installazione.")
    import subprocess
    import sys

    subprocess.check_call([sys.executable, "-m", "pip", "install", "regex"])
    import regex

pre_tokenizer_regex = regex.compile(
    rf"""({special_tokens_pattern})|'s|'t|'re|'ve|'m|'ll|'d| ?\p{{L}}+| ?\p{{N}}+| ?[^\s\p{{L}}\p{{N}}]+|\s+(?!\S)|\s+""")

for text in tqdm(full_corpus):
    words = pre_tokenizer_regex.findall(text)
    for word in words:
        if word:  # Assicuriamoci di non aggiungere stringhe vuote
            word_freqs[word] += 1

# -----------------------------------------------------------------------------
# STEP 3: CREAZIONE DEL VOCABOLARIO INIZIALE
# -----------------------------------------------------------------------------

alphabet = sorted(list(set("".join(word_freqs.keys()))))
# Includiamo i token speciali nel vocabolario iniziale.
# I token speciali non verranno mai uniti o spezzati.
vocab = SPECIAL_TOKENS + [char for char in alphabet if char not in "".join(SPECIAL_TOKENS)]
print(f"Vocabolario iniziale con {len(vocab)} token.")

splits = {word: list(word) for word in word_freqs.keys()}


# -----------------------------------------------------------------------------
# STEP 4: IL CUORE DI BPE - TRAINING LOOP
# -----------------------------------------------------------------------------

def compute_pair_freqs(splits, word_freqs):
    pair_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        # I token speciali non devono essere ulteriormente spezzati o uniti
        if word in SPECIAL_TOKENS:
            continue
        split = splits[word]
        if len(split) < 2:
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            pair_freqs[pair] += freq
    return pair_freqs


def merge_pair(a, b, splits, word_freqs):
    merged_token = a + b
    new_splits = {}
    for word in word_freqs:
        if word in SPECIAL_TOKENS:
            new_splits[word] = [word]
            continue
        split = splits[word]
        i = 0
        new_split = []
        while i < len(split):
            if i < len(split) - 1 and split[i] == a and split[i + 1] == b:
                new_split.append(merged_token)
                i += 2
            else:
                new_split.append(split[i])
                i += 1
        new_splits[word] = new_split
    return new_splits


ordered_merges = []
print(f"3/5 - Addestramento BPE fino a {VOCAB_SIZE} token...")

pbar = tqdm(total=VOCAB_SIZE - len(vocab))
while len(vocab) < VOCAB_SIZE:
    pair_freqs = compute_pair_freqs(splits, word_freqs)
    if not pair_freqs:
        print("\nNessuna coppia da unire. Addestramento terminato prima di raggiungere la dimensione del vocabolario.")
        break

    best_pair = max(pair_freqs, key=pair_freqs.get)

    a, b = best_pair
    splits = merge_pair(a, b, splits, word_freqs)

    ordered_merges.append(best_pair)
    vocab.append(a + b)
    pbar.update(1)

pbar.close()
print("Addestramento BPE completato.")

# -----------------------------------------------------------------------------
# STEP 5: SALVATAGGIO DEL TOKENIZER E CLASSE HELPER
# -----------------------------------------------------------------------------

print("4/5 - Salvataggio del tokenizer in formato JSON...")

token_to_id = {token: i for i, token in enumerate(vocab)}

tokenizer_data = {
    "vocab": token_to_id,
    "merges": ordered_merges,
    "special_tokens": SPECIAL_TOKENS
}

with open(TOKENIZER_FILE, 'w', encoding='utf-8') as f:
    json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)

print(f"Tokenizer salvato in '{TOKENIZER_FILE}'.")


class NanoSocratesTokenizer:
    def __init__(self, tokenizer_path):
        with open(tokenizer_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.vocab = data['vocab']
        self.merges = [tuple(p) for p in data['merges']]
        self.special_tokens = data['special_tokens']
        self.id_to_token = {i: t for t, i in self.vocab.items()}
        self.unk_token_id = self.vocab.get("<UNK>", 0)

        special_tokens_pattern = "|".join(re.escape(token) for token in self.special_tokens)
        self.regex = regex.compile(
            rf"""({special_tokens_pattern})|'s|'t|'re|'ve|'m|'ll|'d| ?\p{{L}}+| ?\p{{N}}+| ?[^\s\p{{L}}\p{{N}}]+|\s+(?!\S)|\s+""")

    def tokenize(self, text):
        pre_tokenized_parts = [word for word in self.regex.findall(text) if word]

        splits = []
        for word in pre_tokenized_parts:
            if word in self.special_tokens:
                splits.append([word])
                continue

            split = list(word)

            for a, b in self.merges:
                i = 0
                new_split = []
                while i < len(split):
                    if i < len(split) - 1 and split[i] == a and split[i + 1] == b:
                        new_split.append(a + b)
                        i += 2
                    else:
                        new_split.append(split[i])
                        i += 1
                split = new_split
            splits.append(split)

        return [token for split in splits for token in split]

    def encode(self, text):
        tokens = self.tokenize(text)
        return [self.vocab.get(token, self.unk_token_id) for token in tokens]

    def decode(self, ids):
        # --- MODIFICA CHIAVE #2: DECODIFICA CORRETTA ---
        # Unisce semplicemente i token. Dal momento che la pre-tokenizzazione
        # ha mantenuto gli spazi come prefissi, questo ricostruirà il testo
        # correttamente.
        return "".join([self.id_to_token.get(id, "<UNK>") for id in ids])


# -----------------------------------------------------------------------------
# ESEMPIO D'USO
# -----------------------------------------------------------------------------
print("\n5/5 - Esempio di utilizzo del tokenizer addestrato:")

try:
    tokenizer = NanoSocratesTokenizer(TOKENIZER_FILE)

    # --- Esempio 1: Task RDF Completion 1 (Masked Language Modeling) ---
    text1 = "<SOT> <SUBJ> dbr:Cabaret_(1972_film) <PRED> <MASK> <OBJ> dbr:Ralph_Burns <EOT>"
    tokens1 = tokenizer.tokenize(text1)
    ids1 = tokenizer.encode(text1)

    print(f"\nTesto Originale 1 (RDF Completion 1): {text1}")
    print(f"Token: {tokens1}")
    print(f"IDs: {ids1}")
    print(f"Testo decodificato: {tokenizer.decode(ids1)}")

    # --- Esempio 2: Task Text2RDF ---
    text2 = "Caddyshack is a 1980 American sports comedy film directed by Harold Ramis <Text2RDF>"
    tokens2 = tokenizer.tokenize(text2)
    ids2 = tokenizer.encode(text2)

    print(f"\nTesto Originale 2 (Text2RDF): {text2}")
    print(f"Token: {tokens2}")
    print(f"IDs: {ids2}")
    print(f"Testo decodificato: {tokenizer.decode(ids2)}")

    # --- Esempio 3: Task RDF2Text ---
    text3 = "<SOT> <SUBJ> dbr:California_(1947_film) <PRED> dbo:director <OBJ> dbr:John_Farrow <EOT> <RDF2Text>"
    tokens3 = tokenizer.tokenize(text3)
    ids3 = tokenizer.encode(text3)

    print(f"\nTesto Originale 3 (RDF2Text): {text3}")
    print(f"Token: {tokens3}")
    print(f"IDs: {ids3}")
    print(f"Testo decodificato: {tokenizer.decode(ids3)}")

    # --- Esempio 4: Task RDF Completion 2 (RDF Generation) ---
    text4 = "<SOT> <SUBJ> dbr:Cain_XVIII <PRED> dbo:imdbId <OBJ> 0176875 <EOT> <CONTINUERDF>"
    tokens4 = tokenizer.tokenize(text4)
    ids4 = tokenizer.encode(text4)

    print(f"\nTesto Originale 4 (RDF Completion 2): {text4}")
    print(f"Token: {tokens4}")
    print(f"IDs: {ids4}")
    print(f"Testo decodificato: {tokenizer.decode(ids4)}")

except FileNotFoundError:
    print(f"\nErrore: Il file del tokenizer '{TOKENIZER_FILE}' non è stato trovato.")
    print("Assicurati di aver eseguito lo script per intero per creare il file.")
except Exception as e:
    import traceback

    print(f"\nSi è verificato un errore inaspettato: {e}")
    traceback.print_exc()