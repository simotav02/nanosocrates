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
VOCAB_SIZE = 16000
CORPUS_FILE = "dataset/training_corpus.txt"
TOKENIZER_FILE = "nanosocrates_tokenizer_1000.json"  # Rinominiamo per chiarezza

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
special_tokens_pattern = f"({'|'.join(regex.escape(token) for token in SPECIAL_TOKENS)})"
special_splitter = regex.compile(special_tokens_pattern)
base_tokenizer_pattern = regex.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+""")

print("2/5 - Pre-tokenizzazione e calcolo delle frequenze...")
for text in tqdm(full_corpus):
    chunks = [chunk for chunk in special_splitter.split(text) if chunk]
    for chunk in chunks:
        if chunk in SPECIAL_TOKENS:
            word_freqs[chunk] += 1
        else:
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
        if word in SPECIAL_TOKENS: continue
        split = splits.get(word)
        if not split or len(split) < 2: continue
        for i in range(len(split) - 1):
            pair_freqs[(split[i], split[i + 1])] += freq
    return pair_freqs


def merge_pair(a, b, splits):
    new_splits = {}
    for word, split in splits.items():
        i, new_split = 0, []
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
    "special_tokens_pattern": special_tokens_pattern
}
with open(TOKENIZER_FILE, 'w', encoding='utf-8') as f:
    json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)
print(f"Tokenizer salvato in '{TOKENIZER_FILE}'.")


class NanoSocratesTokenizer:
    def __init__(self, tokenizer_path):
        with open(tokenizer_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.vocab = data['vocab']
        self.special_tokens = set(data['special_tokens'])
        self.id_to_token = {i: t for t, i in self.vocab.items()}
        self.unk_token_id = self.vocab.get("<UNK>", 0)

        # --- LOGICA CORRETTA ---
        # Convertiamo la lista di merge in un dizionario con la priorità (rank)
        self.merges = {tuple(p): i for i, p in enumerate(data['merges'])}

        self.splitter = regex.compile(data['special_tokens_pattern'])
        self.cache = {}

    def tokenize(self, text):
        final_tokens = []
        # Splitta il testo mantenendo i token speciali come elementi separati
        chunks = [chunk for chunk in self.splitter.split(text) if chunk]

        for chunk in chunks:
            if chunk in self.special_tokens:
                final_tokens.append(chunk)
                continue

            # Per i chunk di testo normale, applichiamo la logica BPE
            if chunk in self.cache:
                final_tokens.extend(self.cache[chunk])
                continue

            # Inizializza la parola come una lista di caratteri
            word_tokens = list(chunk)

            while len(word_tokens) > 1:
                # Trova la prossima coppia da unire con la priorità più alta (rank più basso)
                pairs = {(word_tokens[i], word_tokens[i + 1]): i for i in range(len(word_tokens) - 1)}

                best_pair_to_merge = min(pairs, key=lambda p: self.merges.get(p, float('inf')))

                # Se nessuna delle coppie nella parola è nei nostri merge, abbiamo finito
                if best_pair_to_merge not in self.merges:
                    break

                # Altrimenti, eseguiamo il merge
                idx = pairs[best_pair_to_merge]
                word_tokens = word_tokens[:idx] + [best_pair_to_merge[0] + best_pair_to_merge[1]] + word_tokens[
                                                                                                    idx + 2:]

            self.cache[chunk] = word_tokens
            final_tokens.extend(word_tokens)

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
        ids = tokenizer.encode(text)
        decoded_text = tokenizer.decode(ids)
        print(f"Testo decodificato: {decoded_text}")
        assert decoded_text == text, f"La decodifica non corrisponde all'originale! Test n.{i + 1}"
        print(f"Test {i + 1} superato: Decodifica corretta.")
except FileNotFoundError:
    print(f"\nErrore: Il file del tokenizer '{TOKENIZER_FILE}' non è stato trovato.")
except Exception as e:
    import traceback

    print(f"\nSi è verificato un errore inaspettato: {e}")
    traceback.print_exc()