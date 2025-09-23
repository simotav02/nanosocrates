import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from tokenizers import Tokenizer
from pathlib import Path
from tqdm import tqdm
import warnings
import os
import re

from torchmetrics.text import BLEUScore
from torch.utils.tensorboard import SummaryWriter

# Importa i tuoi moduli custom
from architecture.dataset_lib import NanoSocratesDataset, causal_mask
from architecture.model import build_transformer
from architecture.config import get_config


# Funzioni helper (greedy_decode, parse_rdf_triples) rimangono invariate...
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


def parse_rdf_triples(text: str) -> set:
    triples = set()
    pattern = re.compile(r"<SOT>\s*<SUBJ>\s*(.*?)\s*<PRED>\s*(.*?)\s*<OBJ>\s*(.*?)\s*<EOT>")
    matches = pattern.findall(text)
    for match in matches:
        subj, pred, obj = (m.strip() for m in match)
        if subj and pred and obj:
            triples.add((subj, pred, obj))
    return triples


# <--- MODIFICA 1: La funzione `run_validation` ora accetta un limite di esempi --->
def run_validation(model, validation_ds, tokenizer, max_len, device, global_step, writer, num_examples_to_run):
    model.eval()

    # Liste per accumulare i risultati
    rdf2text_preds, rdf2text_targets = [], []
    mlm_correct, mlm_total = 0, 0
    total_tp, total_fp, total_fn = 0, 0, 0

    # Determina il numero di batch su cui iterare
    desc = "Validating (Quick Check)" if num_examples_to_run > 0 else "Validating (Full)"
    num_batches_to_run = len(validation_ds)
    if num_examples_to_run > 0:
        num_batches_to_run = min(len(validation_ds), num_examples_to_run)

    with torch.no_grad():
        # Usa un contatore per fermarsi se `num_examples_to_run` è impostato
        batch_iterator = tqdm(validation_ds, desc=desc, total=num_batches_to_run)
        count = 0
        for batch in batch_iterator:
            # <--- Logica per fermare il ciclo in anticipo --->
            if num_examples_to_run > 0 and count >= num_examples_to_run:
                break

            # Il resto della logica di validazione rimane identico
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            model_out_tokens = greedy_decode(model, encoder_input, encoder_mask, tokenizer, max_len, device)
            source_text, target_text = batch["src_text"][0], batch["tgt_text"][0]
            model_out_text = tokenizer.decode(model_out_tokens.detach().cpu().numpy(), skip_special_tokens=True)

            if "<RDF2Text>" in source_text:
                rdf2text_preds.append(model_out_text)
                rdf2text_targets.append(target_text)
            elif "<MASK>" in source_text:
                mlm_total += 1
                if model_out_text.strip() == target_text.strip():
                    mlm_correct += 1
            elif "<Text2RDF>" in source_text or "<CONTINUERDF>" in source_text:
                predicted_triples = parse_rdf_triples(tokenizer.decode(model_out_tokens.detach().cpu().numpy()))
                true_triples = parse_rdf_triples(target_text)
                total_tp += len(predicted_triples.intersection(true_triples))
                total_fp += len(predicted_triples.difference(true_triples))
                total_fn += len(true_triples.difference(predicted_triples))

            count += 1

    # Calcolo e logging delle metriche (invariato)
    if writer:
        # ... (il tuo codice di logging qui è già corretto)
        print("-" * 80)
        if rdf2text_preds:
            bleu_metric = BLEUScore()
            bleu = bleu_metric(rdf2text_preds, [[t] for t in rdf2text_targets])
            writer.add_scalar('validation/RDF2Text_BLEU', bleu, global_step)
            print(f"Validation RDF2Text BLEU: {bleu:.4f}")
        if mlm_total > 0:
            accuracy = mlm_correct / mlm_total
            writer.add_scalar('validation/MLM_Accuracy', accuracy, global_step)
            print(f"Validation MLM Accuracy: {accuracy:.4f} ({mlm_correct}/{mlm_total})")
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
    model.train()


# get_ds e get_model rimangono invariati...
def get_ds(config):
    # ...
    return ...


def get_model(config, vocab_size):
    # ...
    return ...


def train_model(config):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    train_dataloader, val_dataloader, tokenizer = get_ds(config)

    # Il resto dell'impostazione (modello, optimizer, ecc.) è invariato...
    model = get_model(config, tokenizer.get_vocab_size()).to(device)
    writer = SummaryWriter(config['experiment_name'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], eps=1e-9)
    total_steps = len(train_dataloader) * config['num_epochs']
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id('<PAD>'), label_smoothing=0.1).to(device)

    global_step = 0
    for epoch in range(config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:
            # Loop di training (invariato)...
            # ...
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        # <--- MODIFICA 2: Esegui la validazione solo se la condizione sull'epoca è soddisfatta --->
        if (epoch + 1) % config['validate_every_n_epochs'] == 0:
            run_validation(
                model=model,
                validation_ds=val_dataloader,
                tokenizer=tokenizer,
                max_len=config['seq_len'],
                device=device,
                global_step=global_step,
                writer=writer,
                num_examples_to_run=config['num_validation_examples']  # <--- Passiamo il nuovo parametro
            )

        # Salvataggio del modello (invariato)
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
    config['experiment_name'] = "runs/nanosocrates_flexible_validation"
    train_model(config)