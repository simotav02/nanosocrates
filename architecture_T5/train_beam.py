# train_lib.py (COMPLETO E DEFINITIVO CON BEAM SEARCH)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tokenizers import Tokenizer
from pathlib import Path
from tqdm import tqdm
import warnings
import os
import re
import evaluate

from torch.utils.tensorboard import SummaryWriter

# Assicurati che questi import puntino ai file corretti
# Se hai rinominato model_lib.py in model.py, questo import è corretto.
from model_lib import build_transformer
from config import get_config
from dataset_lib import NanoSocratesDataset


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


# --- NUOVA FUNZIONE DI DECODIFICA: BEAM SEARCH ---
def beam_search_decode(model, beam_size, source, source_mask, tokenizer, max_len, device):
    sot_idx = tokenizer.token_to_id('<SOT>')
    eot_idx = tokenizer.token_to_id('<EOT>')

    # Pre-calcola l'output dell'encoder (chiamato una sola volta)
    encoder_output = model.encode(source, source_mask)

    # Inizializza il beam. Ogni elemento è una tupla: (sequenza, score)
    # Lo score iniziale è 0.0 (che corrisponde a log(1.0))
    initial_input = torch.empty(1, 1).fill_(sot_idx).type_as(source).to(device)
    beams = [(initial_input, 0.0)]

    completed_beams = []

    for _ in range(max_len - 1):  # Loop al massimo per max_len-1 step
        new_beams = []

        # Flag per interrompere se non ci sono più beam attivi da espandere
        has_active_beams = False

        for candidate_seq, candidate_score in beams:
            # Se un candidato ha già terminato con <EOT>, non espanderlo.
            # Aggiungilo ai risultati finali e passa al prossimo.
            if candidate_seq[0, -1].item() == eot_idx:
                completed_beams.append((candidate_seq, candidate_score))
                continue

            has_active_beams = True

            # --- GESTIONE MASCHERE CORRETTA PER T5 ---
            # Il modello gestisce la causalità internamente. Passiamo None per la maschera del decoder.
            decoder_padding_mask = None

            # Esegui il decoding sul candidato attuale
            out = model.decode(encoder_output, source_mask, candidate_seq, decoder_padding_mask)

            # --- USO CORRETTO DELLE LOG-PROBABILITÀ ---
            # Applica log_softmax per ottenere log-probabilità, matematicamente corretto e stabile
            log_probs = torch.log_softmax(model.project(out[:, -1]), dim=-1)

            # Prendi i top-k token successivi (i più probabili)
            topk_log_probs, topk_idx = torch.topk(log_probs, beam_size, dim=-1)

            # Espandi il beam con i nuovi candidati
            for i in range(beam_size):
                token_idx = topk_idx[0, i].unsqueeze(0).unsqueeze(0)
                token_log_prob = topk_log_probs[0, i].item()

                new_seq = torch.cat([candidate_seq, token_idx], dim=1)
                new_score = candidate_score + token_log_prob  # Somma le log-probabilità

                new_beams.append((new_seq, new_score))

        # Se non ci sono più beam attivi, possiamo interrompere prima
        if not has_active_beams:
            break

        # Ordina tutti i nuovi candidati in base al loro score normalizzato e prendi i migliori `beam_size`
        # La normalizzazione per lunghezza (length penalty) è cruciale per non favorire sequenze troppo corte.
        beams = sorted(new_beams, key=lambda x: x[1] / (x[0].size(1) ** 0.7), reverse=True)[:beam_size]

    # Aggiungi i beam rimasti (che potrebbero non aver raggiunto <EOT>) alla lista dei completati
    completed_beams.extend(beams)

    # Scegli il miglior beam tra tutti quelli completati, usando lo score normalizzato per lunghezza
    best_beam = sorted(completed_beams, key=lambda x: x[1] / (x[0].size(1) ** 0.7), reverse=True)[0]

    return best_beam[0].squeeze()


def parse_rdf_triples(text: str) -> set:
    triples = set()
    pattern = re.compile(r"<SOT>\s*<SUBJ>\s*(.*?)\s*<PRED>\s*(.*?)\s*<OBJ>\s*(.*?)\s*<EOT>")
    matches = pattern.findall(text)
    for match in matches:
        subj, pred, obj = (m.strip() for m in match)
        if subj and pred and obj:
            triples.add((subj, pred, obj))
    return triples


def run_validation(model, validation_ds, tokenizer, max_len, device, global_step, writer, num_examples_to_run):
    model.eval()
    count = -1 if num_examples_to_run == -1 else num_examples_to_run

    rdf2text_preds, rdf2text_labels = [], []
    rdf_gen_tp, rdf_gen_fp, rdf_gen_fn = 0, 0, 0
    mlm_correct, mlm_total = 0, 0
    qualitative_examples = []
    NUM_QUALITATIVE_EXAMPLES = 5
    BEAM_SIZE = 5  # Iperparametro per il beam search

    bleu_metric = evaluate.load("bleu")
    rouge_metric = evaluate.load("rouge")
    meteor_metric = evaluate.load("meteor")

    with torch.no_grad():
        for i, batch in enumerate(tqdm(validation_ds, desc="Validating")):
            if i == count:
                break

            encoder_input = batch["encoder_input"].to(device)
            encoder_padding_mask = batch["encoder_mask"].to(device)

            # --- MODIFICA CHIAVE: USO DEL BEAM SEARCH DECODE ---
            model_out_tokens = beam_search_decode(
                model,
                beam_size=BEAM_SIZE,
                source=encoder_input,
                source_mask=encoder_padding_mask,
                tokenizer=tokenizer,
                max_len=max_len,
                device=device
            )

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]

            model_out_text_clean = tokenizer.decode(model_out_tokens.detach().cpu().numpy(), skip_special_tokens=True)
            model_out_text_raw = tokenizer.decode(model_out_tokens.detach().cpu().numpy(), skip_special_tokens=False)

            if len(qualitative_examples) < NUM_QUALITATIVE_EXAMPLES:
                display_prediction = model_out_text_clean if "<RDF2Text>" in source_text or "<MASK>" in source_text else model_out_text_raw
                qualitative_examples.append(
                    {"source": source_text, "prediction": display_prediction, "ground_truth": target_text})

            if "<RDF2Text>" in source_text:
                if not model_out_text_clean.strip():
                    rdf2text_preds.append(".")
                else:
                    rdf2text_preds.append(model_out_text_clean)
                rdf2text_labels.append([target_text])
            elif "<Text2RDF>" in source_text or "<CONTINUERDF>" in source_text:
                predicted_triples = parse_rdf_triples(model_out_text_raw)
                true_triples = parse_rdf_triples(target_text)
                rdf_gen_tp += len(predicted_triples.intersection(true_triples))
                rdf_gen_fp += len(predicted_triples.difference(true_triples))
                rdf_gen_fn += len(true_triples.difference(predicted_triples))
            elif "<MASK>" in source_text:
                mlm_total += 1
                if model_out_text_clean.strip() == target_text.strip():
                    mlm_correct += 1

    print("\n" + "=" * 80)
    if rdf2text_preds:
        bleu_score = bleu_metric.compute(predictions=rdf2text_preds, references=rdf2text_labels);
        rouge_score = rouge_metric.compute(predictions=rdf2text_preds, references=rdf2text_labels);
        meteor_score = meteor_metric.compute(predictions=rdf2text_preds, references=rdf2text_labels)
        print("--- RDF2Text Metrics ---");
        print(f"BLEU: {bleu_score['bleu']:.4f}");
        print(f"ROUGE-L: {rouge_score['rougeL']:.4f}");
        print(f"METEOR: {meteor_score['meteor']:.4f}\n")
        writer.add_scalar('validation/bleu', bleu_score['bleu'], global_step);
        writer.add_scalar('validation/rougeL', rouge_score['rougeL'], global_step);
        writer.add_scalar('validation/meteor', meteor_score['meteor'], global_step)
    if (rdf_gen_tp + rdf_gen_fp > 0) or (rdf_gen_tp + rdf_gen_fn > 0):
        precision = rdf_gen_tp / (rdf_gen_tp + rdf_gen_fp) if (rdf_gen_tp + rdf_gen_fp) > 0 else 0;
        recall = rdf_gen_tp / (rdf_gen_tp + rdf_gen_fn) if (rdf_gen_tp + rdf_gen_fn) > 0 else 0;
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        print("--- Text2RDF / RDF Completion 2 Metrics ---");
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}\n")
        writer.add_scalar('validation/rdf_f1', f1, global_step);
        writer.add_scalar('validation/rdf_precision', precision, global_step);
        writer.add_scalar('validation/rdf_recall', recall, global_step)
    if mlm_total > 0:
        accuracy = mlm_correct / mlm_total
        print("--- RDF Completion 1 (MLM) Metrics ---");
        print(f"Accuracy: {accuracy:.4f}\n")
        writer.add_scalar('validation/mlm_accuracy', accuracy, global_step)
    print("=" * 80);
    print("--- Esempi Qualitativi ---")
    for idx, example in enumerate(qualitative_examples):
        print(f"\n----- Esempio {idx + 1} -----");
        print(f"INPUT      : {example['source']}");
        print(f"RIFERIMENTO: {example['ground_truth']}");
        print(f"PREDIZIONE : '{example['prediction']}'")
    print("\n" + "=" * 80 + "\n");
    model.train()


def get_ds(config):
    tokenizer = Tokenizer.from_file(str(Path(config['tokenizer_file'])))
    raw_ds = []
    with open(os.path.join(config['data_dir'], "train.source"), 'r', encoding='utf-8') as f_src, open(
            os.path.join(config['data_dir'], "train.target"), 'r', encoding='utf-8') as f_tgt:
        for src_line, tgt_line in zip(f_src, f_tgt):
            raw_ds.append({'source': src_line.strip(), 'target': tgt_line.strip()})
    train_ds_size = int(0.9 * len(raw_ds));
    val_ds_size = len(raw_ds) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(raw_ds, [train_ds_size, val_ds_size])
    train_ds = NanoSocratesDataset(train_ds_raw, tokenizer, config['seq_len']);
    val_ds = NanoSocratesDataset(val_ds_raw, tokenizer, config['seq_len'])
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True);
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=False)
    return train_dataloader, val_dataloader, tokenizer


def get_model(config, vocab_size):
    model = build_transformer(vocab_size=vocab_size, seq_len=config["seq_len"], d_model=config['d_model'],
                              N=config['N'], h=config['h'], dropout=config['dropout'], d_ff=config['d_ff'])
    return model


def train_model(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available() and torch.backends.mps.is_available(): device = "mps"
    print(f"Using device: {device}")
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    train_dataloader, val_dataloader, tokenizer = get_ds(config)
    model = get_model(config, tokenizer.get_vocab_size()).to(device)
    writer = SummaryWriter(config['experiment_name'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], eps=1e-9)
    initial_epoch, global_step = 0, 0
    if config['preload']:
        model_filename = config['preload'];
        print(f"Preloading model {model_filename}");
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict']);
        initial_epoch = state['epoch'] + 1;
        optimizer.load_state_dict(state['optimizer_state_dict']);
        global_step = state['global_step']
        print(f"Resuming training from epoch {initial_epoch}, global step {global_step}")
    num_training_steps = len(train_dataloader) * config['num_epochs'];
    num_warmup_steps = int(num_training_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_training_steps)
    if config['preload']:
        for _ in range(global_step): scheduler.step()
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id('<PAD>'), label_smoothing=0.1).to(device)
    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache();
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:
            optimizer.zero_grad(set_to_none=True)
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_padding_mask = batch['encoder_mask'].to(device)
            decoder_padding_mask = batch['decoder_mask'].to(device)
            encoder_output = model.encode(encoder_input, encoder_padding_mask)
            decoder_output = model.decode(encoder_output, encoder_padding_mask, decoder_input, decoder_padding_mask)
            proj_output = model.project(decoder_output)
            label = batch['label'].to(device)
            loss = loss_fn(proj_output.view(-1, tokenizer.get_vocab_size()), label.view(-1))
            current_lr = optimizer.param_groups[0]['lr']
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}", "lr": f"{current_lr:.2e}"})
            writer.add_scalar('train_loss', loss.item(), global_step);
            writer.add_scalar('learning_rate', current_lr, global_step)
            loss.backward();
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0);
            optimizer.step();
            scheduler.step();
            global_step += 1

        # Esegui validazione e salva il modello
        run_validation(model, val_dataloader, tokenizer, config['seq_len'], device, global_step, writer,
                       config['num_validation_examples'])

        model_filename = f"{config['model_folder']}/{config['model_basename']}{epoch:02d}.pt"
        torch.save(
            {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
             'global_step': global_step}, model_filename)

    writer.close()


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)