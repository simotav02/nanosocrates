# train.py (versione finale con validazione e LR scheduler)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
# <-- NUOVO: Import per lo scheduler -->
from torch.optim.lr_scheduler import CosineAnnealingLR

from tokenizers import Tokenizer
from pathlib import Path
from tqdm import tqdm
import warnings
import os
import torchmetrics
from torch.utils.tensorboard import SummaryWriter

# Importa i tuoi moduli custom
from architecture.dataset_lib import NanoSocratesDataset, causal_mask
from architecture.model import build_transformer
from architecture.config import get_config


# ======================================================================================
# SEZIONE DI VALIDAZIONE (invariata rispetto a prima)
# ======================================================================================

def greedy_decode(model, source, source_mask, tokenizer, max_len, device):
    sot_idx = tokenizer.token_to_id('<SOT>')
    eot_idx = tokenizer.token_to_id('<EOT>')
    encoder_output = model.encode(source, source_mask)
    decoder_input = torch.empty(1, 1).fill_(sot_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)],
                                  dim=1)
        if next_word.item() == eot_idx:
            break
    return decoder_input.squeeze(0)


def run_validation(model, validation_ds, tokenizer, max_len, device, global_step, writer, num_examples=2):
    model.eval()
    count = 0
    source_texts, expected, predicted = [], [], []
    try:
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        console_width = 80
    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"
            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer, max_len, device)
            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer.decode(model_out.detach().cpu().numpy())
            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            print('-' * console_width)
            print(f"{f'SOURCE: ':>12}{source_text}")
            print(f"{f'TARGET: ':>12}{target_text}")
            print(f"{f'PREDICTED: ':>12}{model_out_text}")
            if count == num_examples:
                print('-' * console_width)
                break
    if writer:
        metric = torchmetrics.CharErrorRate()
        writer.add_scalar('validation/cer', metric(predicted, expected), global_step)
        metric = torchmetrics.WordErrorRate()
        writer.add_scalar('validation/wer', metric(predicted, expected), global_step)
        metric = torchmetrics.BLEUScore()
        writer.add_scalar('validation/bleu', metric(predicted, expected), global_step)
        writer.flush()


# ======================================================================================
# SEZIONI get_ds e get_model (invariate)
# ======================================================================================
def get_ds(config):
    # ... (il tuo codice qui, è già corretto)
    print(f"Caricamento tokenizer da: {config['tokenizer_file']}")
    tokenizer = Tokenizer.from_file(config['tokenizer_file'])
    dataset = NanoSocratesDataset(config['corpus_file'], tokenizer, config['seq_len'])
    train_ds_size = int(0.9 * len(dataset))
    val_ds_size = len(dataset) - train_ds_size
    train_ds, val_ds = random_split(dataset, [train_ds_size, val_ds_size])
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)
    return train_dataloader, val_dataloader, tokenizer


def get_model(config, vocab_size):
    # ... (il tuo codice qui, è già corretto)
    return build_transformer(vocab_size, config["seq_len"], d_model=config['d_model'], N=config['N'], h=config['h'],
                             dropout=config['dropout'], d_ff=config['d_ff'])


# ======================================================================================
# SEZIONE DI TRAINING (aggiornata con lo Scheduler)
# ======================================================================================
def train_model(config):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    train_dataloader, val_dataloader, tokenizer = get_ds(config)

    vocab_size = tokenizer.get_vocab_size()
    model = get_model(config, vocab_size).to(device)

    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], eps=1e-9)

    # <-- MODIFICA 1: INIZIALIZZIAMO LO SCHEDULER -->
    # Lo scheduler ridurrà il learning rate da config['lr'] a 0
    # lungo l'intero processo di training.
    # T_max è il numero totale di step di training.
    total_steps = len(train_dataloader) * config['num_epochs']
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    initial_epoch = 0
    global_step = 0

    pad_token_id = tokenizer.token_to_id('<PAD>')
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id, label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")

        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)

            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)

            label = batch['label'].to(device)

            loss = loss_fn(proj_output.view(-1, vocab_size), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            writer.add_scalar('train_loss', loss.item(), global_step)
            # <-- MODIFICA 2: LOGGHIAMO ANCHE IL LEARNING RATE ATTUALE -->
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('learning_rate', current_lr, global_step)
            writer.flush()

            loss.backward()

            # (Opzionale ma consigliato) Gradient Clipping per ulteriore stabilità
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # <-- MODIFICA 3: FACCIAMO AVANZARE LO SCHEDULER -->
            # Questo aggiorna il learning rate ad ogni batch
            scheduler.step()

            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        run_validation(model, val_dataloader, tokenizer, config['seq_len'], device, global_step, writer)

        model_filename = f"{config['model_folder']}/{config['model_basename']}{epoch:02d}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    config['experiment_name'] = "runs/nanosocrates_run_with_scheduler"  # Nuovo nome per non sovrascrivere i log vecchi
    train_model(config)