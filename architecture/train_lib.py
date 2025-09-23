# train.py (Versione completa con TUTTE le metriche richieste dalla traccia)

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

# <--- MODIFICA: Import per le metriche aggiuntive --->
from torchmetrics.text import BLEUScore, ROUGEScore
from torch.utils.tensorboard import SummaryWriter

# Importa i tuoi moduli custom
from architecture.dataset_lib import NanoSocratesDataset, causal_mask
from architecture.model import build_transformer
from architecture.config import get_config

# ======================================================================================
# SEZIONE 1: FUNZIONI HELPER (INVARIATE)
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
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)],
            dim=1
        )
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


# ======================================================================================
# SEZIONE 2: FUNZIONE DI VALIDAZIONE (AGGIORNATA)
# ======================================================================================

def run_validation(model, validation_ds, tokenizer, max_len, device, global_step, writer, num_examples_to_run):
    model.eval()

    rdf2text_preds, rdf2text_targets = [], []
    mlm_correct, mlm_total = 0, 0
    total_tp, total_fp, total_fn = 0, 0, 0

    desc = "Validating (Quick Check)" if num_examples_to_run > 0 else "Validating (Full)"
    num_batches_to_run = len(validation_ds)
    if num_examples_to_run > 0:
        num_batches_to_run = min(len(validation_ds), num_examples_to_run)

    with torch.no_grad():
        batch_iterator = tqdm(validation_ds, desc=desc, total=num_batches_to_run)
        count = 0
        for batch in batch_iterator:
            if num_examples_to_run > 0 and count >= num_examples_to_run:
                break

            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            model_out_tokens = greedy_decode(model, encoder_input, encoder_mask, tokenizer, max_len, device)
            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text_clean = tokenizer.decode(model_out_tokens.detach().cpu().numpy(), skip_special_tokens=True)
            model_out_text_raw = tokenizer.decode(model_out_tokens.detach().cpu().numpy(), skip_special_tokens=False)

            if "<RDF2Text>" in source_text:
                rdf2text_preds.append(model_out_text_clean)
                rdf2text_targets.append(target_text)
            elif "<MASK>" in source_text:
                mlm_total += 1
                if model_out_text_clean.strip() == target_text.strip():
                    mlm_correct += 1
            elif "<Text2RDF>" in source_text or "<CONTINUERDF>" in source_text:
                predicted_triples = parse_rdf_triples(model_out_text_raw)
                true_triples = parse_rdf_triples(target_text)
                total_tp += len(predicted_triples.intersection(true_triples))
                total_fp += len(predicted_triples.difference(true_triples))
                total_fn += len(true_triples.difference(predicted_triples))
            count += 1

    # <--- INIZIO SEZIONE DI CODICE MODIFICATA --- >
    # Calcola e logga le metriche aggregate su TensorBoard
    if writer:
        print("-" * 80)
        # Metriche per RDF2Text
        if rdf2text_preds:
            # Calcolo BLEU (già presente)
            bleu_metric = BLEUScore()
            bleu = bleu_metric(rdf2text_preds, [[t] for t in rdf2text_targets])
            writer.add_scalar('validation/RDF2Text_BLEU', bleu, global_step)
            print(f"Validation RDF2Text BLEU: {bleu:.4f}")

            # <--- NUOVO: Calcolo e logging di ROUGE --->
            # ROUGEScore calcola diverse metriche. Logghiamo la più comune, rougeL_fmeasure.
            rouge_metric = ROUGEScore()
            rouge_scores = rouge_metric(rdf2text_preds, rdf2text_targets)
            rouge_l_f = rouge_scores['rougeL_fmeasure']
            writer.add_scalar('validation/RDF2Text_ROUGE-L', rouge_l_f, global_step)
            print(f"Validation RDF2Text ROUGE-L: {rouge_l_f:.4f}")

            # Calcolo e logging di METEOR
            # try:
            #     meteor_metric = METEOR()
            #     meteor_score = meteor_metric(rdf2text_preds, rdf2text_targets)
            #     writer.add_scalar('validation/RDF2Text_METEOR', meteor_score, global_step)
            #     print(f"Validation RDF2Text METEOR: {meteor_score:.4f}")
            # except Exception as e:
            #     print(f"Could not calculate METEOR score: {e}")

        # Metriche per RDF Completion 1 (MLM) -
        if mlm_total > 0:
            accuracy = mlm_correct / mlm_total
            writer.add_scalar('validation/MLM_Accuracy', accuracy, global_step)
            print(f"Validation MLM Accuracy: {accuracy:.4f} ({mlm_correct}/{mlm_total})")

        # Metriche per Text2RDF e RDF Completion 2 -
        if (total_tp + total_fp) > 0 and (total_tp + total_fn) > 0:
            precision = total_tp / (total_tp + total_fp)
            recall = total_tp / (total_tp + total_fn)
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            writer.add_scalar('validation/RDF_Precision', precision, global_step)
            writer.add_scalar('validation/RDF_Recall', recall, global_step)
            writer.add_scalar('validation/RDF_F1_Score', f1, global_step)
            print(f"Validation RDF Generation F1: {f1:.4f} (P: {precision:.4f}, R: {recall:.4f})")

        print("-" * 80)
        writer.flush()

    model.train()


# ======================================================================================
# SEZIONE 3: FUNZIONI DI SETUP (INVARIATE)
# ======================================================================================

def get_ds(config):
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
    return build_transformer(
        vocab_size,
        config["seq_len"],
        d_model=config['d_model'],
        N=config['N'],
        h=config['h'],
        dropout=config['dropout'],
        d_ff=config['d_ff']
    )


# ======================================================================================
# SEZIONE 4: LOOP DI TRAINING PRINCIPALE (INVARIATO)
# ======================================================================================

def train_model(config):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    train_dataloader, val_dataloader, tokenizer = get_ds(config)
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
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)
            label = batch['label'].to(device)
            loss = loss_fn(proj_output.view(-1, tokenizer.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
            writer.add_scalar('train_loss', loss.item(), global_step)
            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
            writer.flush()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        if (epoch + 1) % config['validate_every_n_epochs'] == 0:
            run_validation(
                model=model,
                validation_ds=val_dataloader,
                tokenizer=tokenizer,
                max_len=config['seq_len'],
                device=device,
                global_step=global_step,
                writer=writer,
                num_examples_to_run=config['num_validation_examples']
            )

        model_filename = f"{config['model_folder']}/{config['model_basename']}{epoch:02d}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)


# ======================================================================================
# SEZIONE 5: ESECUZIONE (INVARIATA)
# ======================================================================================

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    config['experiment_name'] = "runs/nanosocrates_full_metrics"
    train_model(config)