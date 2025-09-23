# train.py (versione con validazione multi-task)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from tokenizers import Tokenizer
from pathlib import Path
from tqdm import tqdm
import warnings
import os
import re  # <--- NUOVO: Import per le espressioni regolari

# <--- NUOVO: Import per le metriche specifiche --->
from torchmetrics.text import BLEUScore
from torchmetrics import Accuracy, Precision, Recall, F1Score

from torch.utils.tensorboard import SummaryWriter

# Importa i tuoi moduli custom
from architecture.dataset_lib import NanoSocratesDataset, causal_mask
from architecture.model import build_transformer
from architecture.config import get_config


# ======================================================================================
# SEZIONE 1: FUNZIONI HELPER PER LA VALUTAZIONE
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


# <--- NUOVA FUNZIONE: Parser per le triple RDF --->
def parse_rdf_triples(text: str) -> set:
    """Estrae un set di triple (soggetto, predicato, oggetto) da una stringa."""
    triples = set()
    # Pattern per trovare <SOT> <SUBJ> ... <PRED> ... <OBJ> ... <EOT>
    pattern = re.compile(r"<SOT>\s*<SUBJ>\s*(.*?)\s*<PRED>\s*(.*?)\s*<OBJ>\s*(.*?)\s*<EOT>")
    matches = pattern.findall(text)
    for match in matches:
        # Rimuoviamo spazi extra per una corrispondenza esatta
        subj = match[0].strip()
        pred = match[1].strip()
        obj = match[2].strip()
        if subj and pred and obj:  # Assicuriamoci che non siano stringhe vuote
            triples.add((subj, pred, obj))
    return triples


# <--- FUNZIONE DI VALIDAZIONE COMPLETAMENTE RISCRITTA --->
def run_validation(model, validation_ds, tokenizer, max_len, device, global_step, writer):
    model.eval()

    # Liste per accumulare i risultati di ogni task
    rdf2text_preds, rdf2text_targets = [], []
    mlm_correct, mlm_total = 0, 0

    # Per Text2RDF e RDF Completion (metrica basata su triple)
    total_tp, total_fp, total_fn = 0, 0, 0

    with torch.no_grad():
        for batch in tqdm(validation_ds, desc="Validating"):
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)

            assert encoder_input.size(0) == 1, "Validation batch size must be 1"

            model_out_tokens = greedy_decode(model, encoder_input, encoder_mask, tokenizer, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            # Decodifichiamo l'output del modello ignorando i token speciali come <PAD>
            model_out_text = tokenizer.decode(model_out_tokens.detach().cpu().numpy(), skip_special_tokens=True)

            # --- Identificazione del task e calcolo della metrica appropriata ---

            if "<RDF2Text>" in source_text:
                # Task: RDF-to-Text -> Usiamo metriche basate su testo (BLEU)
                rdf2text_preds.append(model_out_text)
                rdf2text_targets.append(target_text)

            elif "<MASK>" in source_text:
                # Task: RDF Completion 1 (MLM) -> Usiamo Accuracy (corrispondenza esatta)
                mlm_total += 1
                if model_out_text.strip() == target_text.strip():
                    mlm_correct += 1

            elif "<Text2RDF>" in source_text or "<CONTINUERDF>" in source_text:
                # Task: Text-to-RDF o RDF Completion 2 -> Usiamo Precision/Recall/F1 basate su triple
                predicted_triples = parse_rdf_triples(tokenizer.decode(model_out_tokens.detach().cpu().numpy()))
                true_triples = parse_rdf_triples(target_text)

                tp = len(predicted_triples.intersection(true_triples))
                fp = len(predicted_triples.difference(true_triples))
                fn = len(true_triples.difference(predicted_triples))

                total_tp += tp
                total_fp += fp
                total_fn += fn

    # --- Calcolo e logging delle metriche aggregate per l'intera epoca di validazione ---
    if writer:
        print("-" * 80)
        # Log RDF2Text metrics
        if rdf2text_preds:
            bleu_metric = BLEUScore()
            bleu = bleu_metric(rdf2text_preds,
                               [[t] for t in rdf2text_targets])  # BLEU si aspetta target in una lista di liste
            writer.add_scalar('validation/RDF2Text_BLEU', bleu, global_step)
            print(f"Validation RDF2Text BLEU: {bleu:.4f}")

        # Log MLM metrics
        if mlm_total > 0:
            accuracy = mlm_correct / mlm_total
            writer.add_scalar('validation/MLM_Accuracy', accuracy, global_step)
            print(f"Validation MLM Accuracy: {accuracy:.4f} ({mlm_correct}/{mlm_total})")

        # Log RDF Generation metrics (Text2RDF & ContinueRDF)
        if (total_tp + total_fp) > 0 and (total_tp + total_fn) > 0:
            precision = total_tp / (total_tp + total_fp)
            recall = total_tp / (total_tp + total_fn)
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            writer.add_scalar('validation/RDF_Precision', precision, global_step)
            writer.add_scalar('validation/RDF_Recall', recall, global_step)
            writer.add_scalar('validation/RDF_F1_Score', f1, global_step)
            print(f"Validation RDF Generation F1: {f1:.4f} (P: {precision:.4f}, R: {recall:.4f})")

        print("-" * 80)
        writer.flush()
    model.train()  # Riportiamo il modello in modalità training


# ======================================================================================
# SEZIONI get_ds, get_model E train_model (INVARIATE, ECCETTO PER LA CHIAMATA ALLA VALIDAZIONE)
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


def train_model(config):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    train_dataloader, val_dataloader, tokenizer = get_ds(config)

    vocab_size = tokenizer.get_vocab_size()
    model = get_model(config, vocab_size).to(device)

    writer = SummaryWriter(config['experiment_name'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], eps=1e-9)

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
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('learning_rate', current_lr, global_step)
            writer.flush()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        # <--- La chiamata alla funzione di validazione ora usa la nuova logica --->
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
    config['experiment_name'] = "runs/nanosocrates_multitask_run"
    train_model(config)