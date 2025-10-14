# --- START OF FILE train_final.py ---

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
from transformers import get_linear_schedule_with_warmup

from model_lib import build_transformer
from config_pretrain import get_pretrain_config, get_decoder_tuning_config, get_full_finetune_config, get_full_finetune_config_mla_rope
from dataset_lib import NanoSocratesDataset


# SOSTITUISCI LE VECCHIE FUNZIONI CON QUESTE DUE

def greedy_decode(model, source_or_encoder_output, source_mask, tokenizer, max_len, device,
                  repetition_penalty: float = 1.2):
    sot_idx, eot_idx, pad_idx = tokenizer.token_to_id('<SOT>'), tokenizer.token_to_id('<EOT>'), tokenizer.token_to_id(
        '<PAD>')

    if isinstance(source_or_encoder_output, torch.Tensor) and source_or_encoder_output.dim() == 3:
        encoder_output = source_or_encoder_output
    else:
        encoder_output = model.encode(source_or_encoder_output, source_mask)

    # --- CORREZIONE CRUCIALE ---
    # Il decoder_input deve SEMPRE essere di tipo long perché contiene indici di token.
    # Usiamo .long() invece di .type_as()
    decoder_input = torch.empty(1, 1, dtype=torch.long, device=device).fill_(sot_idx)

    while decoder_input.size(1) < max_len:
        decoder_mask = (decoder_input == pad_idx).to(device)
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        logits = model.projection_layer(out[:, -1])
        if repetition_penalty != 1.0:
            for token_id in set(decoder_input[0].tolist()):
                if logits[0, token_id] > 0:
                    logits[0, token_id] /= repetition_penalty
                else:
                    logits[0, token_id] *= repetition_penalty
        _, next_word = torch.max(logits, dim=1)

        # Anche il nuovo token deve essere di tipo long
        next_word_tensor = torch.empty(1, 1, dtype=torch.long, device=device).fill_(next_word.item())
        decoder_input = torch.cat([decoder_input, next_word_tensor], dim=1)

        if next_word.item() == eot_idx: break
    return decoder_input.squeeze(0)


def beam_search_decode(model, beam_size, source_or_encoder_output, source_mask, tokenizer, max_len, device,
                       repetition_penalty=1.2):
    sot_idx, eot_idx, pad_idx = tokenizer.token_to_id('<SOT>'), tokenizer.token_to_id('<EOT>'), tokenizer.token_to_id(
        '<PAD>')

    if isinstance(source_or_encoder_output, torch.Tensor) and source_or_encoder_output.dim() == 3:
        encoder_output = source_or_encoder_output
    else:
        encoder_output = model.encode(source_or_encoder_output, source_mask)

    # --- CORREZIONE CRUCIALE ---
    # Anche qui, l'input iniziale deve essere di tipo long.
    initial_input = torch.empty(1, 1, dtype=torch.long, device=device).fill_(sot_idx)
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
            logits = model.projection_layer(out[:, -1])
            if repetition_penalty != 1.0:
                for token_id in set(candidate_seq[0].tolist()):
                    if logits[0, token_id] > 0:
                        logits[0, token_id] /= repetition_penalty
                    else:
                        logits[0, token_id] *= repetition_penalty
            log_probs = torch.log_softmax(logits, dim=-1)
            topk_log_probs, topk_idx = torch.topk(log_probs, beam_size, dim=-1)
            for i in range(beam_size):
                # topk_idx è già di tipo long, quindi non serve conversione qui
                token_idx = topk_idx[0, i].unsqueeze(0).unsqueeze(0)
                token_log_prob = topk_log_probs[0, i].item()
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


def parse_rdf_triples_for_strict_eval(text: str) -> set:
    triples = set()
    pattern = re.compile(r"<SOT>\s*<SUBJ>\s*(.*?)\s*<PRED>\s*(.*?)\s*<OBJ>\s*(.*?)\s*<EOT>")
    matches = pattern.findall(text)
    for match in matches:
        subj, pred, obj = (m.strip() for m in match)
        if subj and pred and obj: triples.add((subj, pred, obj))
    return triples


def parse_rdf_triples_for_entity_eval(text: str) -> (list, list, list):
    subjects, predicates, objects = [], [], []
    pattern = re.compile(r"<SOT>\s*<SUBJ>\s*(.*?)\s*<PRED>\s*(.*?)\s*<OBJ>\s*(.*?)\s*<EOT>")
    matches = pattern.findall(text)
    for match in matches:
        subj, pred, obj = (m.strip() for m in match)
        if subj and pred and obj: subjects.append(subj); predicates.append(pred); objects.append(obj)
    return subjects, predicates, objects


def run_validation(model, validation_ds, tokenizer, max_len, device, global_step, writer, num_examples_to_run,
                   decode_strategy_info, phase):
    model.eval()

    if phase == 'pretrain':
        total_val_loss, total_correct_tokens, total_tokens = 0, 0, 0
        loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id('<PAD>')).to(device)
        qualitative_examples = []
        num_qualitative_examples = 5

        with torch.no_grad():
            for i, batch in enumerate(tqdm(validation_ds, desc=f"Validating {phase}")):
                encoder_input, decoder_input = batch['encoder_input'].to(device), batch['decoder_input'].to(device)
                encoder_mask, decoder_mask = batch['encoder_mask'].to(device), batch['decoder_mask'].to(device)
                label = batch['label'].to(device)

                encoder_output = model.encode(encoder_input, encoder_mask)
                decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
                proj_output = model.projection_layer(decoder_output)
                loss = loss_fn(proj_output.view(-1, tokenizer.get_vocab_size()), label.view(-1))
                total_val_loss += loss.item()

                _, predicted_tokens = torch.max(proj_output, dim=-1)
                non_pad_mask = (label != tokenizer.token_to_id('<PAD>'))
                total_correct_tokens += (predicted_tokens.eq(label) & non_pad_mask).sum().item()
                total_tokens += non_pad_mask.sum().item()

                if i < num_qualitative_examples:
                    source_text = batch["src_text"][0]
                    target_text = batch["tgt_text"][0]
                    repetition_penalty = decode_strategy_info.get('repetition_penalty', 1.2)
                    model_out_tokens = greedy_decode(model, encoder_output, encoder_mask, tokenizer, max_len, device,
                                                     repetition_penalty=repetition_penalty)
                    predicted_text = tokenizer.decode(model_out_tokens.cpu().numpy(), skip_special_tokens=False)
                    qualitative_examples.append({
                        "source": source_text,
                        "target": target_text,
                        "predicted": predicted_text
                    })

        avg_val_loss = total_val_loss / len(validation_ds) if validation_ds else 0
        token_accuracy = total_correct_tokens / total_tokens if total_tokens > 0 else 0

        print(f"\n--- {phase.capitalize()} Validation Metrics ---")
        print(f"Average Validation Loss: {avg_val_loss:.4f}")
        print(f"Token-level Accuracy: {token_accuracy:.4f}\n")

        if qualitative_examples:
            print("\n" + "=" * 40 + "\n--- ESEMPI QUALITATIVI DI PRE-TRAINING ---\n" + "=" * 40)
            for ex in qualitative_examples:
                print(f"\nSOURCE:    {ex['source']}")
                print(f"TARGET:    {ex['target']}")
                print(f"PREDICTED: {ex['predicted']}")
            print("\n" + "=" * 40 + "\n")

        if writer:
            writer.add_scalar(f'validation/{phase}/loss', avg_val_loss, global_step)
            writer.add_scalar(f'validation/{phase}/token_accuracy', token_accuracy, global_step)

    elif phase in ['decoder_tune', 'full_finetune']:
        source_texts, expected, predicted_raw, predicted_clean = [], [], [], []

        token_tp, token_fp, token_fn = 0, 0, 0
        pad_id = tokenizer.token_to_id('<PAD>')
        total_val_loss = 0
        loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id).to(device)

        desc = f"Validating {phase.replace('_', '-').capitalize()} ({decode_strategy_info.get('strategy', 'greedy').capitalize()})"
        with torch.no_grad():
            for batch in tqdm(validation_ds, desc=desc):
                encoder_input = batch['encoder_input'].to(device)
                encoder_mask = batch['encoder_mask'].to(device)
                decoder_input = batch['decoder_input'].to(device)
                decoder_mask = batch['decoder_mask'].to(device)
                label = batch['label'].to(device)
                source_text = batch['src_text'][0]
                target_text = batch['tgt_text'][0]
                label_tokens = label[0]

                # --- Calcolo della loss (CORRETTO) ---
                encoder_output = model.encode(encoder_input, encoder_mask)
                decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
                proj_output = model.projection_layer(decoder_output)
                loss = loss_fn(proj_output.view(-1, tokenizer.get_vocab_size()), label.view(-1))
                total_val_loss += loss.item()

                strategy = decode_strategy_info.get('strategy', 'greedy')
                repetition_penalty = decode_strategy_info.get('repetition_penalty', 1.2)

                if strategy == 'beam':
                    beam_size = decode_strategy_info.get('beam_size', 4)
                    model_out_tokens = beam_search_decode(model, beam_size, encoder_output, encoder_mask, tokenizer,
                                                          max_len, device, repetition_penalty)
                else:
                    model_out_tokens = greedy_decode(model, encoder_output, encoder_mask, tokenizer, max_len, device,
                                                     repetition_penalty)

                pred_len = model_out_tokens.size(0)
                true_len = label_tokens[label_tokens != pad_id].size(0)
                max_len_comp = max(pred_len, true_len)
                padded_preds = torch.cat(
                    [model_out_tokens, torch.full((max_len_comp - pred_len,), pad_id, dtype=torch.long, device=device)])
                padded_trues = label_tokens[:max_len_comp]

                is_pred_real = (padded_preds != pad_id)
                is_true_real = (padded_trues != pad_id)

                token_tp += ((padded_preds == padded_trues) & is_true_real).sum().item()
                token_fp += ((padded_preds != padded_trues) & is_pred_real).sum().item()
                token_fn += ((padded_preds != padded_trues) & is_true_real).sum().item()

                predicted_text_raw = tokenizer.decode(model_out_tokens.cpu().numpy(), skip_special_tokens=False)
                predicted_text_clean = tokenizer.decode(model_out_tokens.cpu().numpy(), skip_special_tokens=True)

                source_texts.append(source_text)
                expected.append(target_text)
                predicted_raw.append(predicted_text_raw)
                predicted_clean.append(predicted_text_clean)

        print("\n" + "=" * 50 + f"\n--- METRICHE DI VALIDAZIONE {phase.upper()} ---\n" + "=" * 50)

        avg_val_loss = total_val_loss / len(validation_ds) if validation_ds else 0
        print(f"\n--- Metrica Generale ---")
        print(f"  Average Validation Loss: {avg_val_loss:.4f}")
        if writer: writer.add_scalar(f'validation/{phase}/loss', avg_val_loss, global_step)

        token_precision = token_tp / (token_tp + token_fp) if (token_tp + token_fp) > 0 else 0
        token_recall = token_tp / (token_tp + token_fn) if (token_tp + token_fn) > 0 else 0
        token_f1 = 2 * (token_precision * token_recall) / (token_precision + token_recall) if (
                                                                                                          token_precision + token_recall) > 0 else 0

        print("\n--- Metriche a livello di Token (diagnostica) ---")
        print(f"  Precision: {token_precision:.4f}, Recall: {token_recall:.4f}, F1-Score: {token_f1:.4f}")
        if writer: writer.add_scalar(f'validation/{phase}/token_f1', token_f1, global_step)

        all_pred_subjects, all_true_subjects = [], []
        all_pred_predicates, all_true_predicates = [], []
        all_pred_objects, all_true_objects = [], []

        for src, exp, pred_raw in zip(source_texts, expected, predicted_raw):
            if "<Text2RDF>" in src or "<CONTINUERDF>" in src:
                true_s, true_p, true_o = parse_rdf_triples_for_entity_eval(exp)
                pred_s, pred_p, pred_o = parse_rdf_triples_for_entity_eval(pred_raw)
                all_true_subjects.extend(true_s);
                all_pred_subjects.extend(pred_s)
                all_true_predicates.extend(true_p);
                all_pred_predicates.extend(pred_p)
                all_true_objects.extend(true_o);
                all_pred_objects.extend(pred_o)

        def calculate_entity_metrics(predictions, ground_truths):
            pred_counter, true_counter = Counter(predictions), Counter(ground_truths)
            tp = sum((pred_counter & true_counter).values())
            fp = sum(pred_counter.values()) - tp
            fn = sum(true_counter.values()) - tp
            precision = tp / (tp + fp) if tp + fp > 0 else 0
            recall = tp / (tp + fn) if tp + fn > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
            return precision, recall, f1

        print("\n--- Metriche a livello di Entità (su task RDF) ---")
        p, r, f1 = calculate_entity_metrics(all_pred_subjects, all_true_subjects)
        print(f"  Subjects    | Precision: {p:.4f}, Recall: {r:.4f}, F1-Score: {f1:.4f}")
        p, r, f1 = calculate_entity_metrics(all_pred_predicates, all_true_predicates)
        print(f"  Predicates  | Precision: {p:.4f}, Recall: {r:.4f}, F1-Score: {f1:.4f}")
        p, r, f1 = calculate_entity_metrics(all_pred_objects, all_true_objects)
        print(f"  Objects     | Precision: {p:.4f}, Recall: {r:.4f}, F1-Score: {f1:.4f}")

        tasks = ["Text2RDF", "RDF2Text", "CONTINUERDF", "MLM"]
        qualitative_examples_by_task = {}

        task_data = {task: {'expected': [], 'predicted': [], 'predicted_raw': [], 'source': []} for task in tasks}
        for src, exp, pred_clean, pred_raw in zip(source_texts, expected, predicted_clean, predicted_raw):
            task_name = next((task for task in tasks if f"<{task}>" in src), "Unknown")
            if task_name != "Unknown":
                task_data[task_name]['expected'].append(exp)
                task_data[task_name]['predicted'].append(pred_clean)
                task_data[task_name]['predicted_raw'].append(pred_raw)
                task_data[task_name]['source'].append(src)

        for task, data in task_data.items():
            if not data['expected']: continue
            print(f"\n--- Metriche Task: {task} ({len(data['expected'])} esempi) ---")

            if task == "RDF2Text":
                bleu = evaluate.load("bleu").compute(predictions=data['predicted'],
                                                     references=[[e] for e in data['expected']])
                rouge = evaluate.load("rouge").compute(predictions=data['predicted'], references=data['expected'])
                meteor = evaluate.load("meteor").compute(predictions=data['predicted'], references=data['expected'])
                print(f"  BLEU: {bleu['bleu']:.4f}, ROUGE-L: {rouge['rougeL']:.4f}, METEOR: {meteor['meteor']:.4f}")

            elif task in ["Text2RDF", "CONTINUERDF"]:
                tp_s, p_s, a_s = 0, 0, 0
                for exp, pred_raw in zip(data['expected'], data['predicted_raw']):
                    exp_triples = parse_rdf_triples_for_strict_eval(exp)
                    pred_triples = parse_rdf_triples_for_strict_eval(pred_raw)
                    tp_s += len(exp_triples.intersection(pred_triples))
                    p_s += len(pred_triples)
                    a_s += len(exp_triples)

                precision = tp_s / p_s if p_s > 0 else 0
                recall = tp_s / a_s if a_s > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                print(
                    f"  Precision (Strict): {precision:.4f}, Recall (Strict): {recall:.4f}, F1-Score (Strict): {f1:.4f}")

            # --- INIZIO BLOCCO MODIFICATO: Aggiunta Soft Accuracy per MLM ---
            elif task == "MLM":
                correct_strict = 0
                correct_soft = 0
                for exp, pred in zip(data['expected'], data['predicted']):
                    exp_clean = exp.strip().lower()
                    pred_clean = pred.strip().lower()

                    if exp_clean == pred_clean:
                        correct_strict += 1

                    if exp_clean.startswith('dbo:'):
                        exp_predicate = exp_clean.split(':', 1)[1]
                        if exp_predicate == pred_clean:
                            correct_soft += 1
                    else:
                        if exp_clean == pred_clean:
                            correct_soft += 1

                accuracy_strict = correct_strict / len(data['expected']) if data['expected'] else 0
                accuracy_soft = correct_soft / len(data['expected']) if data['expected'] else 0

                print(f"  Accuracy (Strict): {accuracy_strict:.4f}")
                print(f"  Accuracy (Soft): {accuracy_soft:.4f}")
                if writer:
                    writer.add_scalar(f'validation/{phase}/{task}/accuracy_strict', accuracy_strict, global_step)
                    writer.add_scalar(f'validation/{phase}/{task}/accuracy_soft', accuracy_soft, global_step)
            # --- FINE BLOCCO MODIFICATO ---

            if data['expected']:
                if task not in qualitative_examples_by_task:
                    qualitative_examples_by_task[task] = {
                        "source": data['source'][0],
                        "expected": data['expected'][0],
                        "predicted": data['predicted'][0] if task in ["RDF2Text", "MLM"] else data['predicted_raw'][0]
                    }

        print("\n" + "=" * 50 + "\n--- ESEMPI QUALITATIVI ---\n" + "=" * 50)
        for task, ex in qualitative_examples_by_task.items():
            print(f"\n--- Task: {task} ---")
            print(f"  SOURCE:    {ex['source']}")
            print(f"  EXPECTED:  {ex['expected']}")
            print(f"  PREDICTED: {ex['predicted']}")
        print("\n" + "=" * 50)

    model.train()


def get_ds(config, phase: str):
    tokenizer = Tokenizer.from_file(str(Path(config['tokenizer_file'])))
    source_path = os.path.join(config['data_dir'], "train.source")
    target_path = os.path.join(config['data_dir'], "train.target")
    if not os.path.exists(source_path): raise FileNotFoundError(
        f"File di training non trovati in {config['data_dir']}.")
    with open(source_path, 'r', encoding='utf-8') as f_src, open(target_path, 'r', encoding='utf-8') as f_tgt:
        raw_ds = [{'source': s.strip(), 'target': t.strip()} for s, t in zip(f_src, f_tgt) if s.strip()]

    if phase in ['decoder_tune', 'full_finetune']:
        print(f"\n--- Esecuzione dello Split Stratificato per {phase.upper()} (90/10) ---")

        def get_task_category(item):
            return next(
                (task for task in ["RDF2Text", "Text2RDF", "CONTINUERDF", "MLM"] if f"<{task}>" in item['source']),
                "Unknown")

        grouped_ds = defaultdict(list)
        [grouped_ds[get_task_category(item)].append(item) for item in raw_ds]
        if "Unknown" in grouped_ds:
            unknown_count = len(grouped_ds["Unknown"])
            percentage = (unknown_count / len(raw_ds)) * 100
            print(
                "\n" + "=" * 80 + f"\nATTENZIONE: Trovati {unknown_count} esempi ({percentage:.2f}%) con un task non riconosciuto ('Unknown').\n" + "=" * 80 + "\n")
            if percentage > 1.0: raise ValueError("Percentuale di dati 'Unknown' troppo alta. Rigenerare i dati.")
            del grouped_ds["Unknown"]
        train_raw, val_raw = [], []
        for cat, items in sorted(grouped_ds.items()):
            random.shuffle(items)
            split_point = int(0.9 * len(items))
            train_raw.extend(items[:split_point])
            val_raw.extend(items[split_point:])
            print(f"Categoria '{cat}': {len(items[:split_point])} train / {len(items[split_point:])} val")
    else:
        print(f"\n--- Esecuzione dello Split Casuale Semplice per {phase.capitalize()} (90/10) ---")
        random.shuffle(raw_ds)
        split_point = int(0.9 * len(raw_ds))
        train_raw, val_raw = raw_ds[:split_point], raw_ds[split_point:]

    random.shuffle(train_raw)
    random.shuffle(val_raw)
    print(f"Totale: {len(train_raw)} esempi di training, {len(val_raw)} esempi di validazione.\n")
    train_ds = NanoSocratesDataset(train_raw, tokenizer, config['seq_len'])
    val_ds = NanoSocratesDataset(val_raw, tokenizer, config['seq_len'])
    train_dl = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
    val_batch_size = 1
    val_dl = DataLoader(val_ds, batch_size=val_batch_size, shuffle=False)
    return train_dl, val_dl, tokenizer


def train_model(config, phase: str):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"--- INIZIO FASE: {phase.UPPER()} ---\nUsing device: {device}")
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    train_dataloader, val_dataloader, tokenizer = get_ds(config, phase)

    # --- INIZIO BLOCCO MODIFICATO ---
    # Estraiamo i parametri per il modello dalla configurazione.
    # Assicuriamoci di includere la NUOVA CHIAVE 'attention_type_str'.
    model_params = [
        "d_model", "N", "h", "dropout", "d_ff", "seq_len", "attention_type_str"
    ]
    model_config = {k: v for k, v in config.items() if k in model_params}

    # Ora la configurazione corretta viene passata al costruttore.
    model = build_transformer(
        vocab_size=tokenizer.get_vocab_size(),
        **model_config
    ).to(device)
    # --- FINE BLOCCO MODIFICATO ---

    if config.get("freeze_encoder", False):
        print("\n--- CONGELAMENTO DEI PESI DELL'ENCODER ATTIVO ---")
        for name, param in model.named_parameters():
            if name.startswith('embedding.') or name.startswith('encoder_blocks.') or name.startswith('encoder_norm.'):
                param.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parametri totali: {total_params:,}")
    print(f"Parametri allenabili: {trainable_params:,} ({(trainable_params / total_params) * 100:.2f}%)\n")

    writer = SummaryWriter(config['experiment_name'])
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config['lr'], eps=1e-9,
                                  weight_decay=0.01)

    initial_epoch, global_step = 0, 0
    if config.get('preload'):
        preload_path = config['preload']
        if not os.path.exists(preload_path):
            print(f"ATTENZIONE: File di preload '{preload_path}' non trovato. Il modello partirà da zero.")
        else:
            print(f"Preloading model {preload_path}")
            state = torch.load(preload_path, map_location=device)
            model.load_state_dict(state['model_state_dict'])

            is_new_finetune_phase = phase in ['decoder_tune', 'full_finetune']
            if is_new_finetune_phase and 'pretrain' in preload_path:
                print("Inizio prima fase di fine-tuning (decoder_tune): contatori resettati.")
                initial_epoch, global_step = 0, 0
            elif phase == 'full_finetune' and 'decoder_tuned' in preload_path:
                print("Inizio seconda fase di fine-tuning (full): contatori resettati.")
                initial_epoch, global_step = 0, 0
            else:
                print("Continuazione di un training interrotto.")
                initial_epoch = state.get('epoch', -1) + 1
                global_step = state.get('global_step', 0)
                if 'optimizer_state_dict' in state and not config.get("freeze_encoder", False):
                    optimizer.load_state_dict(state['optimizer_state_dict'])
            print(f"Il training parte dall'epoca {initial_epoch}")

    scheduler_type = config.get('scheduler_type', 'cosine_restarts')
    total_steps = len(train_dataloader) * config['num_epochs']

    if scheduler_type == 'linear_warmup':
        warmup_steps = int(total_steps * config.get('warmup_percentage', 0.1))
        print(f"Scheduler: Linear Warmup + Decay. Total steps: {total_steps}, Warmup steps: {warmup_steps}")
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
    elif scheduler_type == 'cosine_restarts':
        print("Scheduler: CosineAnnealingWarmRestarts")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
    else:
        raise ValueError(f"Scheduler '{scheduler_type}' non riconosciuto.")

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id('<PAD>'),
                                  label_smoothing=config['loss_label_smoothing']).to(device)

    GRADIENT_CLIP_VALUE = 1.0

    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        total_epoch_loss = 0.0
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")

        for batch in batch_iterator:
            optimizer.zero_grad(set_to_none=True)
            encoder_input, decoder_input = batch['encoder_input'].to(device), batch['decoder_input'].to(device)
            encoder_mask, decoder_mask = batch['encoder_mask'].to(device), batch['decoder_mask'].to(device)
            label = batch['label'].to(device)

            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.projection_layer(decoder_output)
            loss = loss_fn(proj_output.view(-1, tokenizer.get_vocab_size()), label.view(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_VALUE)
            optimizer.step()

            if scheduler_type == 'linear_warmup':
                scheduler.step()

            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
            writer.add_scalar(f'train_loss/step/{phase}', loss.item(), global_step)
            total_epoch_loss += loss.item()
            global_step += 1

        avg_epoch_loss = total_epoch_loss / len(train_dataloader)
        print(f"--- Epoch {epoch:02d} finished. Average Training Loss: {avg_epoch_loss:.4f} ---")
        writer.add_scalar(f'train_loss/epoch_avg/{phase}', avg_epoch_loss, epoch)
        writer.add_scalar(f'learning_rate/{phase}', optimizer.param_groups[0]['lr'], epoch)

        if scheduler_type == 'cosine_restarts':
            scheduler.step()

        if (epoch + 1) % config.get('validate_every_n_epochs', 1) == 0:
            print(f"\n--- Running validation for Epoch {epoch:02d} ---")

            decode_strategy_info = {'strategy': 'greedy', 'repetition_penalty': 1.2}
            num_val_examples = config.get('num_validation_examples', -1)

            run_validation(model, val_dataloader, tokenizer, config['seq_len'], device, global_step, writer,
                           num_val_examples, decode_strategy_info, phase)

            torch.save(
                {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                 'global_step': global_step}, f"{config['model_folder']}/{config['model_basename']}{epoch:02d}.pt")

    writer.close()
    print(f"--- FASE {phase.UPPER()} COMPLETATA ---")

    if phase == 'full_finetune':
        print("\n--- RUNNING FINAL EVALUATION WITH BEAM SEARCH ---")
        final_writer = SummaryWriter(config['experiment_name'] + "_final_beam_eval")
        beam_decode_info = {'strategy': 'beam', 'beam_size': 4, 'repetition_penalty': 1.2}
        run_validation(model, val_dataloader, tokenizer, config['seq_len'], device, global_step, final_writer, -1,
                       beam_decode_info, phase)
        final_writer.close()


if __name__ == '__main__':
    warnings.filterwarnings("ignore");
    os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'

    parser = argparse.ArgumentParser(description='Train the NanoSocrates model in phases.')
    parser.add_argument('--phase', type=str, required=True,
                        choices=['pretrain', 'decoder_tune', 'full_finetune'],
                        help="La fase di training da eseguire.")
    args = parser.parse_args()

    if args.phase == 'pretrain':
        config = get_pretrain_config()
    elif args.phase == 'decoder_tune':
        config = get_decoder_tuning_config()
    elif args.phase == 'full_finetune':
        config = get_full_finetune_config_mla_rope()
    else:
        raise ValueError(f"Fase '{args.phase}' non riconosciuta.")

    train_model(config, args.phase)