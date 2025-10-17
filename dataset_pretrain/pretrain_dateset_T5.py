import os
import random
from tokenizers import Tokenizer
from tqdm import tqdm
import numpy as np

# --- Configuration ---
INPUT_CORPUS_FILE = "./pretrain_corpus_data_v3/pretrain_corpus.txt"
TOKENIZER_PATH = "../tokenizer/film_corpus_bpe_tokenizer_t5_3.json"
OUTPUT_DIR = "pretrain_t5_style_data_v3"

# Parameters for the T5 span corruption objective.
CORRUPTION_RATE = 0.15  # Percentage of words to be corrupted.
MEAN_NOISE_SPAN_LENGTH = 3.0  # Average length of a corrupted span.
NUM_EXTRA_ID_TOKENS = 150


def t5_span_corruption_whole_word(text: str, tokenizer: Tokenizer, noise_density: float, mean_noise_span_length: float):
    """
    Implements a T5-style span corruption objective that operates on space-separated "words".
    This prevents the fragmentation of RDF entities and words during masking.

    The process is:
    1. Split the text into a list of words.
    2. Determine which word spans to mask based on noise_density.
    3. Reconstruct the input string, replacing masked spans with sentinel tokens (<extra_id_X>).
    4. Reconstruct the target string, which contains the sentinel tokens followed by the original words.
    """

    # 1. Operate on a word-level to preserve entities.
    words = text.split(' ')
    n_words = len(words)

    if n_words < 2:
        return text, ""  # Not enough words to corrupt.

    # 2. Calculate the number and length of spans to corrupt.
    num_noise_words = int(round(n_words * noise_density))
    num_noise_words = max(1, min(num_noise_words, n_words - 1))

    num_spans = int(round(num_noise_words / mean_noise_span_length))
    num_spans = max(1, num_spans)

    # Generate span lengths from a Poisson distribution.
    span_lengths = np.random.poisson(lam=mean_noise_span_length, size=num_spans)

    # Normalize span lengths to match the target noise density.
    total_generated_length = np.sum(span_lengths)
    if total_generated_length == 0:
        return text, ""
    span_lengths = (span_lengths * (num_noise_words / total_generated_length)).astype(int)
    span_lengths = span_lengths[span_lengths > 0]
    if len(span_lengths) == 0:
        return text, ""

    # Choose unique starting positions for each span.
    possible_start_indices = np.arange(
        n_words - np.max(span_lengths) + 1 if n_words > np.max(span_lengths) else n_words)
    if len(possible_start_indices) == 0:
        return text, ""
    start_indices = np.random.choice(possible_start_indices, size=len(span_lengths), replace=False)

    # Collect all individual word indices that will be masked.
    masked_indices = set()
    for start, length in zip(start_indices, span_lengths):
        masked_indices.update(range(start, min(start + length, n_words)))
    if not masked_indices:
        return text, ""

    # 3. Build the final input and target strings.
    extra_id_tokens = [f"<extra_id_{i}>" for i in range(NUM_EXTRA_ID_TOKENS)]
    max_sentinels_for_spans = NUM_EXTRA_ID_TOKENS - 1

    final_input_words = []
    target_words_list = []
    extra_id_counter = 0

    # Iterate through the words, replacing spans of masked words with sentinel tokens.
    i = 0
    while i < n_words:
        if i not in masked_indices:
            final_input_words.append(words[i])
            i += 1
            continue

        # A span of masked words starts here.
        start_of_span = i
        while i < n_words and i in masked_indices:
            i += 1
        end_of_span = i
        span_original_words = words[start_of_span:end_of_span]

        if extra_id_counter >= max_sentinels_for_spans:
            # If we run out of sentinel tokens, append the original words instead.
            final_input_words.extend(span_original_words)
            continue

        # Replace the span with a sentinel token in the input.
        sentinel_token = extra_id_tokens[extra_id_counter]
        final_input_words.append(sentinel_token)

        # Add the sentinel and the original words to the target.
        target_words_list.append(sentinel_token)
        target_words_list.extend(span_original_words)
        extra_id_counter += 1

    if not target_words_list:
        return text, ""

    # Add the final sentinel token to the end of the target string.
    final_sentinel_token = extra_id_tokens[extra_id_counter]
    target_words_list.append(final_sentinel_token)

    # Rejoin the lists of words back into strings.
    corrupted_text = ' '.join(final_input_words)
    target_text = ' '.join(target_words_list)

    return corrupted_text, target_text


def main():
    """
    Main script to generate the T5-style pre-training dataset.
    It reads a clean corpus, applies the whole-word span corruption to each line,
    and saves the results into parallel 'train.source' and 'train.target' files.
    """
    print(f"--- Creazione del Dataset di Pre-training (Whole Word Masking v5 - Corretto) ---")

    if not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError(
            f"Tokenizer non trovato in '{TOKENIZER_PATH}'. Esegui prima lo script di tokenizzazione.")
    if not os.path.exists(INPUT_CORPUS_FILE):
        raise FileNotFoundError(
            f"Corpus puro non trovato in '{INPUT_CORPUS_FILE}'. Esegui prima create_pretrain_corpus.py.")

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

    with open(INPUT_CORPUS_FILE, 'r', encoding='utf-8') as f:
        corpus_lines = [line.strip() for line in f if line.strip()]

    # Apply the corruption function to each line of the corpus.
    corrupted_examples = []
    for line in tqdm(corpus_lines, desc="Corrompendo il corpus"):
        corrupted_input, target = t5_span_corruption_whole_word(line, tokenizer, CORRUPTION_RATE,
                                                                MEAN_NOISE_SPAN_LENGTH)
        if corrupted_input and target:
            corrupted_examples.append({"input": corrupted_input, "output": target})

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    source_out_path = os.path.join(OUTPUT_DIR, "train.source")
    target_out_path = os.path.join(OUTPUT_DIR, "train.target")

    # Shuffle the dataset before writing to files.
    random.shuffle(corrupted_examples)

    # Write the corrupted inputs and targets to separate files.
    with open(source_out_path, 'w', encoding='utf-8') as f_source, \
            open(target_out_path, 'w', encoding='utf-8') as f_target:
        for example in tqdm(corrupted_examples, desc="Scrivendo i file"):
            f_source.write(example['input'] + '\n')
            f_target.write(example['output'] + '\n')

    print(f"\nPROCESSO COMPLETATO.")
    print(f"Creati {len(corrupted_examples)} esempi di pre-training.")
    print(f"File salvati in '{OUTPUT_DIR}/train.source' e '{OUTPUT_DIR}/train.target'.")


if __name__ == "__main__":
    main()
