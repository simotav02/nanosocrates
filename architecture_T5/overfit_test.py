import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
from tqdm import tqdm
import os

from dataset_lib import NanoSocratesDataset
from model_lib import build_transformer
from config import get_config


NUM_STEPS_FOR_TEST = 300
BATCH_SIZE_FOR_TEST = 4
LEARNING_RATE_FOR_TEST = 3e-4
LABEL_SMOOTHING_FOR_TEST = 0.0  .


def get_single_batch(config):
    tokenizer = Tokenizer.from_file(config['tokenizer_file'])

    # Carica il dataset completo (potremmo ottimizzare leggendo solo le prime righe)
    source_path = os.path.join(config['data_dir'], "train.source")
    target_path = os.path.join(config['data_dir'], "train.target")
    with open(source_path, 'r', encoding='utf-8') as f:
        source_lines = f.readlines()
    with open(target_path, 'r', encoding='utf-8') as f:
        target_lines = f.readlines()

    full_raw_ds = [{'source': src.strip(), 'target': tgt.strip()} for src, tgt in zip(source_lines, target_lines)]

    if len(full_raw_ds) < BATCH_SIZE_FOR_TEST:
        raise ValueError(f"Il dataset non ha abbastanza esempi per creare un batch di dimensione {BATCH_SIZE_FOR_TEST}")

    full_dataset = NanoSocratesDataset(full_raw_ds, tokenizer, config['seq_len'])
    temp_dataloader = DataLoader(full_dataset, batch_size=BATCH_SIZE_FOR_TEST, shuffle=False)

    single_batch = next(iter(temp_dataloader))

    print(f"Estratto un singolo batch di {BATCH_SIZE_FOR_TEST} esempi per il test di overfitting.")
    return single_batch, tokenizer


def main():
    print("--- INIZIO SANITY CHECK: OVERFITTING SU UN SINGOLO BATCH ---")
    print(f"LR={LEARNING_RATE_FOR_TEST}, Label Smoothing={LABEL_SMOOTHING_FOR_TEST}, Steps={NUM_STEPS_FOR_TEST}")
    print("-" * 60)

    config = get_config()
    config['preload'] = None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available() and torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")

    single_batch, tokenizer = get_single_batch(config)

    model = build_transformer(
        vocab_size=tokenizer.get_vocab_size(),
        seq_len=config['seq_len'],
        d_model=config['d_model'],
        N=config['N'],
        h=config['h'],
        dropout=config['dropout'],
        d_ff=config['d_ff']
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE_FOR_TEST, eps=1e-9)

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id('<PAD>'),
                                  label_smoothing=LABEL_SMOOTHING_FOR_TEST).to(device)

    encoder_input = single_batch['encoder_input'].to(device)
    decoder_input = single_batch['decoder_input'].to(device)
    encoder_mask = single_batch['encoder_mask'].to(device)
    decoder_mask = single_batch['decoder_mask'].to(device)
    label = single_batch['label'].to(device)

    pbar = tqdm(range(NUM_STEPS_FOR_TEST), desc="Overfitting a single batch")
    for step in pbar:
        model.train()

        encoder_output = model.encode(encoder_input, encoder_mask)
        decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
        proj_output = model.project(decoder_output)

        loss = loss_fn(proj_output.view(-1, tokenizer.get_vocab_size()), label.view(-1))

        pbar.set_postfix({"loss": f"{loss.item():6.4f}"})

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    final_loss = loss.item()
    print(f"\nFine del test. Loss finale: {final_loss:.6f}")

    print("-" * 60)
    print("--- SANITY CHECK COMPLETATO ---")

    if final_loss < 0.01:
        print("✅ TEST SUPERATO: La loss è scesa a un valore vicino allo zero.")
    else:
        print(
            "❌ TEST FALLITO: La loss non è scesa a sufficienza. Potrebbe esserci un problema nel modello o nel training loop.")


if __name__ == "__main__":
    main()