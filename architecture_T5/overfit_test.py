import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tokenizers import Tokenizer
from tqdm import tqdm
import os

from dataset_lib import NanoSocratesDataset
from model import build_transformer
from config import get_config

# --- CONFIGURAZIONE DEL TEST ---
NUM_SAMPLES_FOR_TEST = 30
NUM_EPOCHS_FOR_TEST = 100
BATCH_SIZE_FOR_TEST = 4
LEARNING_RATE_FOR_TEST = 5e-4
LABEL_SMOOTHING_FOR_TEST = 0.0  # DISABILITATO! Cruciale per l'overfitting.


def get_overfit_ds(config, num_samples: int):
    tokenizer = Tokenizer.from_file(config['tokenizer_file'])
    source_path = os.path.join(config['data_dir'], "train.source")
    target_path = os.path.join(config['data_dir'], "train.target")
    with open(source_path, 'r', encoding='utf-8') as f:
        source_lines = f.readlines()
    with open(target_path, 'r', encoding='utf-8') as f:
        target_lines = f.readlines()
    full_raw_ds = [{'source': src.strip(), 'target': tgt.strip()} for src, tgt in zip(source_lines, target_lines)]
    if len(full_raw_ds) < num_samples:
        raise ValueError(
            f"Richiesti {num_samples} esempi per il test, ma il dataset ne contiene solo {len(full_raw_ds)}")
    full_dataset = NanoSocratesDataset(full_raw_ds, tokenizer, config['seq_len'])
    overfit_subset = Subset(full_dataset, range(num_samples))
    overfit_dataloader = DataLoader(overfit_subset, batch_size=BATCH_SIZE_FOR_TEST, shuffle=True)
    print(f"Creato un mini-dataset per il test con {len(overfit_subset)} esempi.")
    return overfit_dataloader, tokenizer


def main():
    print("--- INIZIO SANITY CHECK DI OVERFITTING (con iperparametri aggressivi) ---")
    print(f"LR={LEARNING_RATE_FOR_TEST}, Label Smoothing={LABEL_SMOOTHING_FOR_TEST}")
    print("-" * 50)

    config = get_config()
    config['num_epochs'] = NUM_EPOCHS_FOR_TEST
    config['batch_size'] = BATCH_SIZE_FOR_TEST
    config['preload'] = None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available() and torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")

    train_dataloader, tokenizer = get_overfit_ds(config, NUM_SAMPLES_FOR_TEST)

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

    for epoch in range(config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Overfitting Epoch {epoch:02d}/{config['num_epochs']}")
        total_loss = 0
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)

            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)

            label = batch['label'].to(device)
            loss = loss_fn(proj_output.view(-1, tokenizer.get_vocab_size()), label.view(-1))

            current_lr = optimizer.param_groups[0]['lr']
            batch_iterator.set_postfix({"loss": f"{loss.item():6.4f}", "lr": f"{current_lr:.2e}"})
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        avg_loss = total_loss / len(train_dataloader)
        print(f"Fine Epoch {epoch:02d} - Average Loss: {avg_loss:.4f}")

    print("-" * 50)
    print("--- SANITY CHECK COMPLETATO ---")

if __name__ == "__main__":
    main()