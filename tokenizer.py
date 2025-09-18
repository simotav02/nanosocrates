import re
import json
from collections import defaultdict
from tqdm import tqdm  # Per una bella barra di progresso

# -----------------------------------------------------------------------------
# STEP 1: CONFIGURAZIONE E DEFINIZIONE DEI TOKEN SPECIALI
# -----------------------------------------------------------------------------

# Come da traccia, definiamo tutti i token speciali necessari.
# Aggiungiamo anche <PAD> (per il padding) e <UNK> (per token sconosciuti),
# che sono fondamentali per l'addestramento del modello.
SPECIAL_TOKENS = [
    "<PAD>", "<UNK>", "<SOT>", "<EOT>", "<SUBJ>", "<PRED>", "<OBJ>",
    "<RDF2Text>", "<Text2RDF>", "<CONTINUERDF>", "<MASK>"
]

# Rispettiamo l'hint del professore: usiamo un vocabolario ridotto.
# 16000 è un buon punto di partenza.
VOCAB_SIZE = 16000

# Il percorso del file generato nello Step 2
CORPUS_FILE = "training_corpus.txt"

# Il file dove salveremo il nostro tokenizer addestrato
TOKENIZER_FILE = "nanosocrates_tokenizer.json"

# -----------------------------------------------------------------------------
# STEP 2: PRE-TOKENIZZAZIONE E CALCOLO DELLE FREQUENZE
# -----------------------------------------------------------------------------

# Leggiamo il corpus. Il file è in formato input \t output.
# Per addestrare il tokenizer, usiamo entrambi.
print("1/5 - Lettura e preparazione del corpus...")
full_corpus = []
with open(CORPUS_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        # Aggiungiamo sia l'input che l'output al corpus per il training
        parts = line.strip().split('\t')
        for part in parts:
            full_corpus.append(part)

# Calcoliamo la frequenza di ogni "parola".
# La pre-tokenizzazione è cruciale. Usiamo una regex che:
# 1. Isola i nostri token speciali (es. <SOT>).
# 2. Separa le parole (es. "Inception").
# 3. Separa i numeri e la punteggiatura.
# Questa logica è simile a quella usata dai tokenizer moderni.
word_freqs = defaultdict(int)
# Questa regex cattura i token speciali O sequenze di lettere O sequenze di numeri O qualsiasi altro carattere singolo.
pre_tokenizer_regex = r"(<SOT>|<EOT>|<SUBJ>|<PRED>|<OBJ>|<RDF2Text>|<Text2RDF>|<CONTINUERDF>|<MASK>|'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^a-zA-Z0-9\s]+|\s)"

print("2/5 - Pre-tokenizzazione e calcolo delle frequenze...")
for text in tqdm(full_corpus):
    # Applichiamo la pre-tokenizzazione
    words = re.findall(pre_tokenizer_regex, text)
    for word in words:
        word_freqs[word] += 1

# -----------------------------------------------------------------------------
# STEP 3: CREAZIONE DEL VOCABOLARIO INIZIALE
# -----------------------------------------------------------------------------

# Il vocabolario iniziale è composto dai token speciali e da tutti i singoli
# caratteri presenti nel nostro corpus.
alphabet = []
for word in word_freqs.keys():
    for letter in word:
        if letter not in alphabet:
            alphabet.append(letter)
alphabet.sort()

# Il nostro vocabolario parte con i token speciali, seguiti dall'alfabeto.
vocab = SPECIAL_TOKENS + alphabet
print(f"Vocabolario iniziale con {len(vocab)} token.")

# Inizializziamo le "parole" come sequenze di caratteri, come nell'esempio HF.
splits = {word: list(word) for word in word_freqs.keys()}


# -----------------------------------------------------------------------------
# STEP 4: IL CUORE DI BPE - TRAINING LOOP
# -----------------------------------------------------------------------------

# Questa è l'implementazione "from scratch" della logica di BPE.

def compute_pair_freqs(splits, word_freqs):
    """Calcola la frequenza di ogni coppia di token adiacenti."""
    pair_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        split = splits[word]
        if len(split) < 2:
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            pair_freqs[pair] += freq
    return pair_freqs


def merge_pair(a, b, splits, word_freqs):
    """Fonde una coppia di token (a, b) in un nuovo token (ab) in tutti i 'splits'."""
    merged_token = a + b
    new_splits = {}
    for word in word_freqs:
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


# Addestriamo il tokenizer
merges = {}
print(f"3/5 - Addestramento BPE fino a {VOCAB_SIZE} token...")

pbar = tqdm(total=VOCAB_SIZE - len(vocab))
while len(vocab) < VOCAB_SIZE:
    pair_freqs = compute_pair_freqs(splits, word_freqs)
    if not pair_freqs:
        print("Nessuna coppia da unire. Addestramento terminato.")
        break

    # Troviamo la coppia più frequente
    best_pair = max(pair_freqs, key=pair_freqs.get)

    # La uniamo
    a, b = best_pair
    splits = merge_pair(a, b, splits, word_freqs)

    # Salviamo la regola di unione e aggiungiamo il nuovo token al vocabolario
    merged_token = a + b
    merges[best_pair] = merged_token
    vocab.append(merged_token)
    pbar.update(1)

pbar.close()
print("Addestramento BPE completato.")

# -----------------------------------------------------------------------------
# STEP 5: SALVATAGGIO DEL TOKENIZER E CLASSE HELPER
# -----------------------------------------------------------------------------

print("4/5 - Salvataggio del tokenizer in formato JSON...")

# Creiamo i dizionari per la conversione token <-> id
token_to_id = {token: i for i, token in enumerate(vocab)}
id_to_token = {i: token for token, i in enumerate(vocab)}

# Salviamo tutto in un unico file JSON
tokenizer_data = {
    "vocab": token_to_id,
    "merges": {f"{p[0]} {p[1]}": m for p, m in merges.items()},  # json non supporta tuple come chiavi
    "special_tokens": SPECIAL_TOKENS
}

with open(TOKENIZER_FILE, 'w', encoding='utf-8') as f:
    json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)

print(f"Tokenizer salvato in '{TOKENIZER_FILE}'.")


# Una classe per rendere facile l'utilizzo del nostro tokenizer
class NanoSocratesTokenizer:
    def __init__(self, tokenizer_path):
        with open(tokenizer_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.vocab = data['vocab']
        # Ricostruiamo le tuple dalle chiavi stringa
        self.merges = {tuple(k.split()): v for k, v in data['merges'].items()}
        self.special_tokens = data['special_tokens']

        self.id_to_token = {i: t for t, i in self.vocab.items()}
        self.unk_token_id = self.vocab.get("<UNK>", 0)

    def tokenize(self, text):
        # 1. Pre-tokenizzazione
        words = re.findall(pre_tokenizer_regex, text)

        # 2. Dividi ogni parola nei suoi caratteri
        splits = [list(word) for word in words]

        # 3. Applica iterativamente le regole di unione apprese
        for pair, merged_token in self.merges.items():
            a, b = pair
            for i, split in enumerate(splits):
                j = 0
                new_split = []
                while j < len(split):
                    if j < len(split) - 1 and split[j] == a and split[j + 1] == b:
                        new_split.append(merged_token)
                        j += 2
                    else:
                        new_split.append(split[j])
                        j += 1
                splits[i] = new_split

        # 4. Concatena i risultati
        return [token for split in splits for token in split]

    def encode(self, text):
        tokens = self.tokenize(text)
        return [self.vocab.get(token, self.unk_token_id) for token in tokens]

    def decode(self, ids):
        tokens = [self.id_to_token.get(id, "<UNK>") for id in ids]
        return "".join(tokens).replace(' ', ' ').strip()


# -----------------------------------------------------------------------------
# ESEMPIO D'USO (AGGIORNATO CON ESEMPI DAL TUO CORPUS)
# -----------------------------------------------------------------------------
print("\n5/5 - Esempio di utilizzo del tokenizer addestrato:")

# Carichiamo il tokenizer appena addestrato
tokenizer = NanoSocratesTokenizer(TOKENIZER_FILE)

# --- Esempio 1: Task RDF Completion 1 (Masked Language Modeling) ---
# Preso direttamente dal tuo training_corpus.txt
text1 = "<SOT> <SUBJ> dbr:Cabaret_(1972_film) <PRED> <MASK> <OBJ> dbr:Ralph_Burns <EOT>"
tokens1 = tokenizer.tokenize(text1)
ids1 = tokenizer.encode(text1)

print(f"\nTesto Originale 1 (RDF Completion 1): {text1}")
print(f"Token: {tokens1}")
print(f"IDs: {ids1}")
print(f"Testo decodificato: {tokenizer.decode(ids1)}")

# --- Esempio 2: Task Text2RDF ---
# Preso direttamente dal tuo training_corpus.txt
text2 = "Caddyshack is a 1980 American sports comedy film directed by Harold Ramis <Text2RDF>"
tokens2 = tokenizer.tokenize(text2)
ids2 = tokenizer.encode(text2)

print(f"\nTesto Originale 2 (Text2RDF): {text2}")
print(f"Token: {tokens2}")
print(f"IDs: {ids2}")
print(f"Testo decodificato: {tokenizer.decode(ids2)}")

# --- Esempio 3: Task RDF2Text ---
# Preso direttamente dal tuo training_corpus.txt
text3 = "<SOT> <SUBJ> dbr:California_(1947_film) <PRED> dbo:director <OBJ> dbr:John_Farrow <EOT> <RDF2Text>"
tokens3 = tokenizer.tokenize(text3)
ids3 = tokenizer.encode(text3)

print(f"\nTesto Originale 3 (RDF2Text): {text3}")
print(f"Token: {tokens3}")
print(f"IDs: {ids3}")
print(f"Testo decodificato: {tokenizer.decode(ids3)}")

# --- Esempio 4: Task RDF Completion 2 (RDF Generation) ---
# Preso direttamente dal tuo training_corpus.txt
text4 = "<SOT> <SUBJ> dbr:Cain_XVIII <PRED> dbo:imdbId <OBJ> 0176875 <EOT> <CONTINUERDF>"
tokens4 = tokenizer.tokenize(text4)
ids4 = tokenizer.encode(text4)

print(f"\nTesto Originale 4 (RDF Completion 2): {text4}")
print(f"Token: {tokens4}")
print(f"IDs: {ids4}")
print(f"Testo decodificato: {tokenizer.decode(ids4)}")