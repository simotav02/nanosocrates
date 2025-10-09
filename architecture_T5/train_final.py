# train_final.py (Script Unico - Versione Definitiva Corretta)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
from pathlib import Path
from tqdm import tqdm
import warnings
import os
import re
import evaluate
import argparse
import random
from collections import defaultdict, Counter

from torch.utils.tensorboard import SummaryWriter

from model_lib import build_transformer
from config_pretrain import get_pretrain_config, get_task_adapt_config, get_finetune_config
from dataset_lib import NanoSocratesDataset


# --- FUNZIONI DI UTILITY (decodifica, parse) ---
def greedy_decode(model, source, source_mask, tokenizer, max_len, device):
    sot_idx = tokenizer.token_to_id('<SOT>')
    eot_idx = tokenizer.token_to_id('<EOT>')
    pad_idx = tokenizer.token_to_id('<PAD>')
    encoder_output = model.encode(source, source_mask)
    decoder_input = torch.empty(1, 1).fill_(sot_idx).type_as(source).to(device)
    while decoder_input.size(1) < max_len:
        decoder_mask = (decoder_input == pad_idx).to(device)
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)],
                                  dim=1)
        if next_word.item() == eot_idx: break
    return decoder_input.squeeze(0)


def beam_search_decode(model, beam_size, source, source_mask, tokenizer, max_len, device, repetition_penalty=1.2):
    sot_idx, eot_idx, pad_idx = tokenizer.token_to_id('<SOT>'), tokenizer.token_to_id('<EOT>'), tokenizer.token_to_id(
        '<PAD>')
    encoder_output = model.encode(source, source_mask)
    initial_input = torch.empty(1, 1).fill_(sot_idx).type_as(source).to(device)
    beams = [(initial_input, 0.0)]
    completed_beams = []
    for _ in range(max_len - 1):
        new_beams, has_active_beams = [], False
        for candidate_seq, candidate_score in beams:
            if candidate_seq[0, -1].item() == eot_idx:
                completed_beams.append((candidate_seq, candidate_score));
                continue
            has_active_beams = True
            decoder_mask = (candidate_seq == pad_idx).to(device)
            out = model.decode(encoder_output, source_mask, candidate_seq, decoder_mask)
            logits = model.project(out[:, -1])

            # --- APPLICAZIONE REPETITION PENALTY ---
            if repetition_penalty != 1.0:
                for token_id in set(candidate_seq[0].tolist()):
                    if logits[0, token_id] > 0:
                        logits[0, token_id] /= repetition_penalty
                    else:
                        logits[0, token_id] *= repetition_penalty

            log_probs = torch.log_softmax(logits, dim=-1)
            topk_log_probs, topk_idx = torch.topk(log_probs, beam_size, dim=-1)
            for i in range(beam_size):
                token_idx, token_log_prob = topk_idx[0, i].unsqueeze(0).unsqueeze(0), topk_log_probs[0, i].item()
                new_seq = torch.cat([candidate_seq, token_idx], dim=1)
                new_beams.append((new_seq, candidate_score + token_log_prob))
        if not has_active_beams: break
        beams = sorted(new_beams, key=lambda x: x[1] / (x[0].size(1) ** 0.7), reverse=True)[:beam_size]
        if all(b[0][0, -1].item() == eot_idx for b in beams):
            completed_beams.extend(beams);
            break
    if not completed_beams: completed_beams = beams
    best_beam = sorted(completed_beams, key=lambda x: x[1] / (x[0].size(1) ** 0.7), reverse=True)[0]
    return best_beam[0].squeeze()


def top_k_sampling_decode(model, source, source_mask, tokenizer, max_len, device, k=50, repetition_penalty=1.2):
    sot_idx, eot_idx, pad_idx = tokenizer.token_to_id('<SOT>'), tokenizer.token_to_id('<EOT>'), tokenizer.token_to_id(
        '<PAD>')
    encoder_output = model.encode(source, source_mask)
    decoder_input = torch.empty(1, 1).fill_(sot_idx).type_as(source).to(device)
    while decoder_input.size(1) < max_len:
        decoder_mask = (decoder_input == pad_idx).to(device)
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        logits = model.project(out[:, -1])

        # --- APPLICAZIONE REPETITION PENALTY ---
        if repetition_penalty != 1.0:
            for token_id in set(decoder_input[0].tolist()):
                if logits[0, token_id] > 0:
                    logits[0, token_id] /= repetition_penalty
                else:
                    logits[0, token_id] *= repetition_penalty

        top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)
        probabilities = torch.softmax(top_k_logits, dim=-1)
        next_token_id = top_k_indices.gather(-1, torch.multinomial(probabilities, num_samples=1))
        decoder_input = torch.cat([decoder_input, next_token_id], dim=1)
        if next_token_id.item() == eot_idx: break
    return decoder_input.squeeze(0)


def parse_rdf_triples(text: str) -> set:
    triples = set()
    pattern = re.compile(r"<SOT>\s*<SUBJ>\s*(.*?)\s*<PRED>\s*(.*?)\s*<OBJ>\s*(.*?)\s*<EOT>")
    matches = pattern.findall(text)
    for match in matches:
        subj, pred, obj = (m.strip() for m in match)
        if subj and pred and obj: triples.add((subj, pred, obj))
    return triples


def run_validation(model, validation_ds, tokenizer, max_len, device, global_step, writer, num_examples_to_run,
                   decode_strategy, phase):
    model.eval()
    if phase == 'pretrain' or phase == 'task_adapt':
        total_val_loss, total_correct_tokens, total_tokens = 0, 0, 0
        loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id('<PAD>')).to(device)
        with torch.no_grad():
            for batch in tqdm(validation_ds, desc=f"Validating {phase}"):
                encoder_input, decoder_input = batch['encoder_input'].to(device), batch['decoder_input'].to(device)
                encoder_mask, decoder_mask = batch['encoder_mask'].to(device), batch['decoder_mask'].to(device)
                label = batch['label'].to(device)
                encoder_output = model.encode(encoder_input, encoder_mask)
                decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
                proj_output = model.project(decoder_output)
                loss = loss_fn(proj_output.view(-1, tokenizer.get_vocab_size()), label.view(-1))
                total_val_loss += loss.item()
                _, predicted_tokens = torch.max(proj_output, dim=-1)
                non_pad_mask = (label != tokenizer.token_to_id('<PAD>'))
                total_correct_tokens += (predicted_tokens.eq(label) & non_pad_mask).sum().item()
                total_tokens += non_pad_mask.sum().item()
        avg_val_loss = total_val_loss / len(validation_ds) if len(validation_ds) > 0 else 0
        token_accuracy = total_correct_tokens / total_tokens if total_tokens > 0 else 0
        print(
            f"\n--- {phase.capitalize()} Validation Metrics ---\nAverage Validation Loss: {avg_val_loss:.4f}\nToken-level Accuracy: {token_accuracy:.4f}\n")
        if writer:
            writer.add_scalar(f'validation/{phase}/loss', avg_val_loss, global_step)
            writer.add_scalar(f'validation/{phase}/token_accuracy', token_accuracy, global_step)
        model.train()
        return

    rdf2text_preds, rdf2text_labels = [], []
    text2rdf_tp, text2rdf_fp, text2rdf_fn = 0, 0, 0
    continuerdf_tp, continuerdf_fp, continuerdf_fn = 0, 0, 0
    mlm_correct, mlm_total = 0, 0
    qualitative_examples = {}
    tasks_needed = {"Text2RDF", "RDF2Text", "MLM", "CONTINUERDF"}
    task_counter = Counter()
    sot_id, eot_id = tokenizer.token_to_id('<SOT>'), tokenizer.token_to_id('<EOT>')

    with torch.no_grad():
        desc = f"Validating ({decode_strategy.capitalize()})"
        for i, batch in enumerate(tqdm(validation_ds, desc=desc)):
            if i == num_examples_to_run: break
            encoder_input, encoder_padding_mask = batch["encoder_input"].to(device), batch["encoder_mask"].to(device)
            if decode_strategy == 'beam':
                model_out_tokens = beam_search_decode(model, 5, encoder_input, encoder_padding_mask, tokenizer, max_len,
                                                      device)
            else:
                model_out_tokens = top_k_sampling_decode(model, encoder_input, encoder_padding_mask, tokenizer, max_len,
                                                         device)

            source_text, target_text = batch["src_text"][0], batch["tgt_text"][0]
            tokens_to_decode = model_out_tokens.detach().cpu().numpy()
            model_out_text_raw = tokenizer.decode(tokens_to_decode, skip_special_tokens=False)
            start_index = 1 if len(tokens_to_decode) > 0 and tokens_to_decode[0] == sot_id else 0
            eot_indices = [idx for idx, token_id in enumerate(tokens_to_decode) if token_id == eot_id]
            end_index = eot_indices[0] if eot_indices else len(tokens_to_decode)
            model_out_text_clean = tokenizer.decode(tokens_to_decode[start_index:end_index],
                                                    skip_special_tokens=True).strip()
            current_task = next((task for task in tasks_needed if f"<{task}>" in source_text), None)
            if current_task and current_task not in qualitative_examples:
                prediction_to_show = model_out_text_raw if current_task in ["Text2RDF",
                                                                            "CONTINUERDF"] else model_out_text_clean
                qualitative_examples[current_task] = {"source": source_text, "prediction": prediction_to_show,
                                                      "ground_truth": target_text}
            if current_task: task_counter[current_task] += 1
            if current_task == "RDF2Text":
                rdf2text_preds.append(model_out_text_clean or ".");
                rdf2text_labels.append([target_text])
            elif current_task in ["Text2RDF", "CONTINUERDF"]:
                predicted_triples, true_triples = parse_rdf_triples(model_out_text_raw), parse_rdf_triples(target_text)
                tp, fp, fn = len(predicted_triples.intersection(true_triples)), len(
                    predicted_triples.difference(true_triples)), len(true_triples.difference(predicted_triples))
                if current_task == "Text2RDF":
                    text2rdf_tp += tp; text2rdf_fp += fp; text2rdf_fn += fn
                else:
                    continuerdf_tp += tp; continuerdf_fp += fp; continuerdf_fn += fn
            elif current_task == "MLM":
                mlm_total += 1
                if model_out_text_clean == target_text.strip(): mlm_correct += 1

    print(f"\n" + "=" * 80 + f"\nRiepilogo task trovati nel validation set: {dict(task_counter)}")
    if rdf2text_preds:
        bleu = evaluate.load("bleu").compute(predictions=rdf2text_preds, references=rdf2text_labels);
        rouge = evaluate.load("rouge").compute(predictions=rdf2text_preds, references=rdf2text_labels);
        meteor = evaluate.load("meteor").compute(predictions=rdf2text_preds, references=rdf2text_labels)
        print(
            f"--- RDF2Text Metrics ---\nBLEU: {bleu['bleu']:.4f}, ROUGE-L: {rouge['rougeL']:.4f}, METEOR: {meteor['meteor']:.4f}\n")
    for task_name, (tp, fp, fn) in [("Text2RDF", (text2rdf_tp, text2rdf_fp, text2rdf_fn)),
                                    ("CONTINUERDF", (continuerdf_tp, continuerdf_fp, continuerdf_fn))]:
        if (tp + fp > 0) or (tp + fn > 0):
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0;
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0;
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            print(
                f"--- {task_name} Metrics ---\nPrecision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}\n")
    if mlm_total > 0:
        accuracy = mlm_correct / mlm_total
        print(f"--- RDF Completion (MLM) Metrics ---\nAccuracy: {accuracy:.4f}\n")
    print("=" * 80 + "\n--- Esempi Qualitativi (Uno per Task) ---")
    for task_name in sorted(list(tasks_needed)):
        ex = qualitative_examples.get(task_name)
        print(f"\n--- Esempio Task: {task_name} ---")
        if ex:
            print(f"INPUT      : {ex['source']}\nRIFERIMENTO: {ex['ground_truth']}\nPREDIZIONE : '{ex['prediction']}'")
        else:
            print("Nessun esempio trovato in questo batch di validazione.")
    print("\n" + "=" * 80 + "\n");
    model.train()


def get_ds(config, phase: str):
    tokenizer = Tokenizer.from_file(str(Path(config['tokenizer_file'])))
    source_path = os.path.join(config['data_dir'], "train.source");
    target_path = os.path.join(config['data_dir'], "train.target")
    if not os.path.exists(source_path): raise FileNotFoundError(
        f"File di training non trovati in {config['data_dir']}.")
    with open(source_path, 'r', encoding='utf-8') as f_src, open(target_path, 'r', encoding='utf-8') as f_tgt:
        raw_ds = [{'source': s.strip(), 'target': t.strip()} for s, t in zip(f_src, f_tgt) if s.strip()]

    if phase == 'finetune':
        print("\n--- Esecuzione dello Split Stratificato per Fine-Tuning (90/10) ---")

        def get_task_category(item):
            return next(
                (task for task in ["RDF2Text", "Text2RDF", "CONTINUERDF", "MLM"] if f"<{task}>" in item['source']),
                "Unknown")

        grouped_ds = defaultdict(list);
        [grouped_ds[get_task_category(item)].append(item) for item in raw_ds]
        train_raw, val_raw = [], [];
        [(train_raw.extend(items[:int(0.9 * len(items))]), val_raw.extend(items[int(0.9 * len(items)):]), print(
            f"Categoria '{cat}': {len(items[:int(0.9 * len(items))])} train / {len(items[int(0.9 * len(items)):])} val"))
         for cat, items in sorted(grouped_ds.items()) if random.shuffle(items) is None]
    else:
        print(f"\n--- Esecuzione dello Split Casuale Semplice per {phase.capitalize()} (90/10) ---")
        random.shuffle(raw_ds);
        split_point = int(0.9 * len(raw_ds));
        train_raw, val_raw = raw_ds[:split_point], raw_ds[split_point:]

    random.shuffle(train_raw);
    random.shuffle(val_raw)
    print(f"Totale: {len(train_raw)} esempi di training, {len(val_raw)} esempi di validazione.\n")
    train_ds = NanoSocratesDataset(train_raw, tokenizer, config['seq_len'])
    val_ds = NanoSocratesDataset(val_raw, tokenizer, config['seq_len'])
    train_dl = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
    val_batch_size = config['batch_size'] if phase != 'finetune' else 1
    val_dl = DataLoader(val_ds, batch_size=val_batch_size, shuffle=False)
    return train_dl, val_dl, tokenizer


def train_model(config, phase: str):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"--- INIZIO FASE: {phase.upper()} ---\nUsing device: {device}")
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    train_dataloader, val_dataloader, tokenizer = get_ds(config, phase)

    model_config = {k: v for k, v in config.items() if k in ["d_model", "N", "h", "dropout", "d_ff", "seq_len"]}
    model = build_transformer(vocab_size=tokenizer.get_vocab_size(), **model_config).to(device)
    writer = SummaryWriter(config['experiment_name'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], eps=1e-9, weight_decay=0.01)

    initial_epoch, global_step = 0, 0
    if config.get('preload'):
        preload_path = config['preload']
        if not os.path.exists(preload_path):
            print(f"ATTENZIONE: File di preload '{preload_path}' non trovato. Il modello partirà da zero.")
        else:
            print(f"Preloading model {preload_path}")
            state = torch.load(preload_path, map_location=device)
            model.load_state_dict(state['model_state_dict'])

            is_new_phase = (phase == 'task_adapt' and 'pretrain' in preload_path) or \
                           (phase == 'finetune' and ('task_adapt' in preload_path or 'pretrain' in preload_path))

            if is_new_phase:
                print(f"Inizio di una nuova fase ({phase}): i contatori di epoca e step vengono resettati.")
                initial_epoch, global_step = 0, 0
            else:
                print("Continuazione di un training interrotto.")
                initial_epoch = state.get('epoch', -1) + 1
                if 'optimizer_state_dict' in state:
                    optimizer.load_state_dict(state['optimizer_state_dict'])
                global_step = state.get('global_step', 0)

            print(f"Il training parte dall'epoca {initial_epoch}")

    if phase == 'pretrain':
        print("Scheduler: CosineAnnealingWarmRestarts con T_0=10 e T_mult=2")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    else:  # Per task_adapt e finetune
        # print(f"Scheduler: LinearLR con decadimento lineare per {config['num_epochs']} epoche.")
        # scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1,
        #                                               total_iters=config['num_epochs'])
        print("Scheduler: CosineAnnealingWarmRestarts con T_0=10 e T_mult=2")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id('<PAD>'),
                                  label_smoothing=config['loss_label_smoothing']).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        total_loss = 0.0
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:
            optimizer.zero_grad(set_to_none=True)
            encoder_input, decoder_input = batch['encoder_input'].to(device), batch['decoder_input'].to(device)
            encoder_mask, decoder_mask = batch['encoder_mask'].to(device), batch['decoder_mask'].to(device)
            label = batch['label'].to(device)
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)
            loss = loss_fn(proj_output.view(-1, tokenizer.get_vocab_size()), label.view(-1))
            loss.backward()
            optimizer.step()
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
            writer.add_scalar(f'train_step_loss/{phase}', loss.item(), global_step)
            total_loss += loss.item()
            global_step += 1

        avg_epoch_loss = total_loss / len(batch_iterator)
        writer.add_scalar(f'train_epoch_loss/{phase}', avg_epoch_loss, epoch)
        print(f"--- Epoch {epoch:02d} finished. Average Training Loss: {avg_epoch_loss:.4f} ---")
        scheduler.step()
        writer.add_scalar(f'learning_rate/{phase}', optimizer.param_groups[0]['lr'], epoch)
        print(f"Learning rate per la prossima epoca: {optimizer.param_groups[0]['lr']:.2e}")

        if (epoch + 1) % config.get('validate_every_n_epochs', 1) == 0:
            print(f"\n--- Running validation for Epoch {epoch:02d} ---")
            run_validation(model, val_dataloader, tokenizer, config['seq_len'], device, global_step, writer,
                           config['num_validation_examples'], 'sampling', phase)
            torch.save(
                {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                 'global_step': global_step}, f"{config['model_folder']}/{config['model_basename']}{epoch:02d}.pt")

    writer.close()
    print(f"--- FASE {phase.upper()} COMPLETATA ---")

    if phase == 'finetune':
        print("\n--- RUNNING FINAL EVALUATION WITH BEAM SEARCH ---")
        final_writer = SummaryWriter(config['experiment_name'] + "_final_beam_eval")
        run_validation(model, val_dataloader, tokenizer, config['seq_len'], device, global_step, final_writer, -1,
                       'beam', phase)
        final_writer.close()


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
    parser = argparse.ArgumentParser(description='Train the NanoSocrates model in phases.')
    parser.add_argument('--phase', type=str, required=True, choices=['pretrain', 'task_adapt', 'finetune'])
    args = parser.parse_args()
    if args.phase == 'pretrain':
        config = get_pretrain_config()
    elif args.phase == 'task_adapt':
        config = get_task_adapt_config()
    else:
        config = get_finetune_config()
    train_model(config, args.phase)