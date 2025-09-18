import json
from collections import defaultdict
from tqdm import tqdm

# Tenta di importare il modulo 'regex'. Se non esiste, lo installa.
try:
    import regex
except ImportError:
    print("Modulo 'regex' non trovato. Tentativo di installazione...")
    import subprocess
    import sys

    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "regex"])
        import regex

        print("Modulo 'regex' installato con successo.")
    except Exception as e:
        print(
            f"Impossibile installare il modulo 'regex'. Per favore, installalo manualmente con 'pip install regex'. Errore: {e}")
        sys.exit(1)

# -----------------------------------------------------------------------------
# STEP 1: CONFIGURAZIONE
# -----------------------------------------------------------------------------

SPECIAL_TOKENS = [
    "<PAD>", "<UNK>", "<SOT>", "<EOT>", "<SUBJ>", "<PRED>", "<OBJ>",
    "<RDF2Text>", "<Text2RDF>", "<CONTINUERDF>", "<MASK>"
]
VOCAB_SIZE = 34000
CORPUS_FILE = "../dataset/training_corpus_1000.txt"
TOKENIZER_FILE = "nanosocrates_tokenizer_1000.json"

# -----------------------------------------------------------------------------
# STEP 2: PRE-TOKENIZZAZIONE E CALCOLO DELLE FREQUENZE (LOGICA RIVISTA)
# -----------------------------------------------------------------------------

print("1/5 - Lettura e preparazione del corpus...")
full_corpus = []
with open(CORPUS_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        for part in parts:
            full_corpus.append(part)

word_freqs = defaultdict(int)

# Pattern per splittare MANTENENDO i token speciali
special_tokens_pattern = f"({'|'.join(regex.escape(token) for token in SPECIAL_TOKENS)})"
special_splitter = regex.compile(special_tokens_pattern)

# Pattern per tokenizzare il testo "normale" (tutto ciò che non è un token speciale)
base_tokenizer_pattern = regex.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+""")

print("2/5 - Pre-tokenizzazione e calcolo delle frequenze...")
for text in tqdm(full_corpus):
    # Prima splittiamo per i token speciali
    chunks = [chunk for chunk in special_splitter.split(text) if chunk]
    for chunk in chunks:
        if chunk in SPECIAL_TOKENS:
            # Se il chunk è un token speciale, contiamo quello
            word_freqs[chunk] += 1
        else:
            # Altrimenti, tokenizziamo il chunk di testo normale
            for match in base_tokenizer_pattern.finditer(chunk):
                word_freqs[match.group(0)] += 1

# -----------------------------------------------------------------------------
# STEP 3: CREAZIONE DEL VOCABOLARIO INIZIALE
# -----------------------------------------------------------------------------

alphabet = sorted(list(set("".join(word_freqs.keys()))))
vocab = SPECIAL_TOKENS + [char for char in alphabet if char not in "".join(SPECIAL_TOKENS)]
print(f"Vocabolario iniziale con {len(vocab)} token.")

splits = {word: list(word) for word in word_freqs.keys() if word not in SPECIAL_TOKENS}


# -----------------------------------------------------------------------------
# STEP 4: IL CUORE DI BPE - TRAINING LOOP
# -----------------------------------------------------------------------------

def compute_pair_freqs(splits, word_freqs):
    pair_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        if word in SPECIAL_TOKENS:
            continue
        split = splits.get(word)
        if not split or len(split) < 2:
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            pair_freqs[pair] += freq
    return pair_freqs


def merge_pair(a, b, splits):
    new_splits = {}
    for word, split in splits.items():
        i = 0
        new_split = []
        while i < len(split):
            if i < len(split) - 1 and split[i] == a and split[i + 1] == b:
                new_split.append(a + b)
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
    splits = merge_pair(a, b, splits)
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
    "special_tokens": SPECIAL_TOKENS,
    "special_tokens_pattern": special_tokens_pattern,
    "base_tokenizer_pattern": base_tokenizer_pattern.pattern
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
        self.special_tokens = set(data['special_tokens'])
        self.id_to_token = {i: t for t, i in self.vocab.items()}
        self.unk_token_id = self.vocab.get("<UNK>", 0)
        self.special_splitter = regex.compile(data['special_tokens_pattern'])
        self.base_tokenizer = regex.compile(data['base_tokenizer_pattern'])
        self.cache = {}

    def tokenize(self, text):
        final_tokens = []
        chunks = [chunk for chunk in self.special_splitter.split(text) if chunk]

        for chunk in chunks:
            if chunk in self.special_tokens:
                final_tokens.append(chunk)
                continue

            # Pre-tokenize del chunk normale
            words = [match.group(0) for match in self.base_tokenizer.finditer(chunk)]

            for word in words:
                if word in self.cache:
                    final_tokens.extend(self.cache[word])
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

                self.cache[word] = split
                final_tokens.extend(split)
        return final_tokens

    def encode(self, text):
        return [self.vocab.get(token, self.unk_token_id) for token in self.tokenize(text)]

    def decode(self, ids):
        return "".join([self.id_to_token.get(id, "<UNK>") for id in ids])


# -----------------------------------------------------------------------------
# ESEMPIO D'USO
# -----------------------------------------------------------------------------
print("\n5/5 - Esempio di utilizzo del tokenizer addestrato:")
try:
    tokenizer = NanoSocratesTokenizer(TOKENIZER_FILE)

    texts_to_test = [
        "<SOT> <SUBJ> dbr:Cabaret_(1972_film) <PRED> <MASK> <OBJ> dbr:Ralph_Burns <EOT>",
        "Caddyshack is a 1980 American sports comedy film directed by Harold Ramis <Text2RDF>",
        "<SOT> <SUBJ> dbr:California_(1947_film) <PRED> dbo:director <OBJ> dbr:John_Farrow <EOT> <RDF2Text>",
        "<SOT> <SUBJ> dbr:Cain_XVIII <PRED> dbo:imdbId <OBJ> 0176875 <EOT> <CONTINUERDF>"
    ]

    for i, text in enumerate(texts_to_test):
        print(f"\n--- Test {i + 1} ---")
        print(f"Testo Originale: {text}")

        tokens = tokenizer.tokenize(text)
        print(f"Token: {tokens}")

        decoded_text = tokenizer.decode(tokenizer.encode(text))
        print(f"Testo decodificato: {decoded_text}")

        assert decoded_text == text, "La decodifica non corrisponde all'originale!"
        print("Test superato: Decodifica corretta.")

except FileNotFoundError:
    print(f"\nErrore: Il file del tokenizer '{TOKENIZER_FILE}' non è stato trovato.")
except Exception as e:
    import traceback

    print(f"\nSi è verificato un errore inaspettato: {e}")
    traceback.print_exc()