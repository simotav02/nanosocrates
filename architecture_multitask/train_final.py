# train_final.py

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
from config_pretrain import get_pretrain_config, get_finetune_config
from dataset_lib import NanoSocratesDataset


def greedy_decode(model, source, source_mask, tokenizer, max_len, device, repetition_penalty: float = 1.2):
    sot_idx, eot_idx, pad_idx = tokenizer.token_to_id('<SOT>'), tokenizer.token_to_id('<EOT>'), tokenizer.token_to_id(
        '<PAD>')
    encoder_output = model.encode(source, source_mask)
    decoder_input = torch.empty(1, 1).fill_(sot_idx).type_as(source).to(device)
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


def parse_rdf_triples_for_entity_eval(text: str) -> (list, list, list):
    subjects, predicates, objects = [], [], []
    pattern = re.compile(r"<SOT>\s*<SUBJ>\s*(.*?)\s*<PRED>\s*(.*?)\s*<OBJ>\s*(.*?)\s*<EOT>")
    matches = pattern.findall(text)
    for match in matches:
        subj, pred, obj = (m.strip() for m in match)
        if subj and pred and obj: subjects.append(subj); predicates.append(pred); objects.append(obj)
    return subjects, predicates, objects


def parse_rdf_triples_for_strict_eval(text: str) -> set:
    triples = set()
    pattern = re.compile(r"<SOT>\s*<SUBJ>\s*(.*?)\s*<PRED>\s*(.*?)\s*<OBJ>\s*(.*?)\s*<EOT>")
    matches = pattern.findall(text)
    for match in matches:
        subj, pred, obj = (m.strip() for m in match)
        if subj and pred and obj: triples.add((subj, pred, obj))
    return triples


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
                    model_out_tokens = greedy_decode(model, encoder_input, encoder_mask, tokenizer, max_len, device,
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


    elif phase == 'finetune':
        source_texts, expected, predicted = [], [], []

        bleu_metric = evaluate.load('bleu')
        rouge_metric = evaluate.load('rouge')
        meteor_metric = evaluate.load('meteor')

        count = 0
        desc = f"Validating Finetune ({decode_strategy_info.get('strategy', 'greedy').capitalize()})"
        with torch.no_grad():
            for batch in tqdm(validation_ds, desc=desc):
                count += 1
                if num_examples_to_run > 0 and count > num_examples_to_run: break

                encoder_input = batch['encoder_input'].to(device)
                encoder_mask = batch['encoder_mask'].to(device)
                source_text = batch['src_text'][0]
                target_text = batch['tgt_text'][0]

                strategy = decode_strategy_info.get('strategy', 'greedy')
                repetition_penalty = decode_strategy_info.get('repetition_penalty', 1.2)

                if strategy == 'beam':
                    beam_size = decode_strategy_info.get('beam_size', 4)
                    model_out_tokens = beam_search_decode(model, beam_size, encoder_input, encoder_mask, tokenizer,
                                                          max_len, device, repetition_penalty)
                else:
                    model_out_tokens = greedy_decode(model, encoder_input, encoder_mask, tokenizer, max_len, device,
                                                     repetition_penalty)

                predicted_text = tokenizer.decode(model_out_tokens.cpu().numpy(), skip_special_tokens=True)

                source_texts.append(source_text)
                expected.append(target_text)
                predicted.append(predicted_text)

        tasks = ["Text2RDF", "RDF2Text", "CONTINUERDF", "MLM"]
        task_metrics = {task: defaultdict(list) for task in tasks}
        qualitative_examples_by_task = {task: [] for task in tasks}
        num_qualitative_per_task = 2

        for src, exp, pred in zip(source_texts, expected, predicted):
            task_name = next((task for task in tasks if f"<{task}>" in src), "Unknown")
            if task_name != "Unknown":
                task_metrics[task_name]['expected'].append(exp)
                task_metrics[task_name]['predicted'].append(pred)
                if len(qualitative_examples_by_task[task_name]) < num_qualitative_per_task:
                    qualitative_examples_by_task[task_name].append({"source": src, "expected": exp, "predicted": pred})

        print("\n" + "=" * 50 + "\n--- METRICHE DI VALIDAZIONE FINETUNE ---\n" + "=" * 50)
        for task, data in task_metrics.items():
            if not data['expected']: continue

            print(f"\n--- Task: {task} ({len(data['expected'])} esempi) ---")

            if task == "RDF2Text":
                bleu = bleu_metric.compute(predictions=data['predicted'], references=[[e] for e in data['expected']])
                rouge = rouge_metric.compute(predictions=data['predicted'], references=data['expected'])
                meteor = meteor_metric.compute(predictions=data['predicted'], references=data['expected'])
                print(f"  BLEU: {bleu['bleu']:.4f}, ROUGE-L: {rouge['rougeL']:.4f}, METEOR: {meteor['meteor']:.4f}")
                if writer:
                    writer.add_scalar(f'validation/{phase}/{task}/bleu', bleu['bleu'], global_step)
                    writer.add_scalar(f'validation/{phase}/{task}/rougeL', rouge['rougeL'], global_step)
                    writer.add_scalar(f'validation/{phase}/{task}/meteor', meteor['meteor'], global_step)

            elif task in ["Text2RDF", "CONTINUERDF"]:
                tp_s, p_s, a_s = 0, 0, 0
                for exp, pred in zip(data['expected'], data['predicted']):
                    exp_triples = parse_rdf_triples_for_strict_eval(exp)
                    pred_triples = parse_rdf_triples_for_strict_eval(pred)
                    tp_s += len(exp_triples.intersection(pred_triples))
                    p_s += len(pred_triples)
                    a_s += len(exp_triples)

                precision = tp_s / p_s if p_s > 0 else 0
                recall = tp_s / a_s if a_s > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                print(
                    f"  Precision (Strict): {precision:.4f}, Recall (Strict): {recall:.4f}, F1-Score (Strict): {f1:.4f}")
                if writer:
                    writer.add_scalar(f'validation/{phase}/{task}/precision', precision, global_step)
                    writer.add_scalar(f'validation/{phase}/{task}/recall', recall, global_step)
                    writer.add_scalar(f'validation/{phase}/{task}/f1', f1, global_step)

            elif task == "MLM":
                correct = sum(1 for exp, pred in zip(data['expected'], data['predicted']) if
                              exp.strip().lower() == pred.strip().lower())
                accuracy = correct / len(data['expected']) if data['expected'] else 0
                print(f"  Accuracy (Strict): {accuracy:.4f}")
                if writer:
                    writer.add_scalar(f'validation/{phase}/{task}/accuracy', accuracy, global_step)

        print("\n--- Esempi Qualitativi di Fine-Tuning ---")
        for task, examples in qualitative_examples_by_task.items():
            if not examples: continue
            print(f"\n--- Task: {task} ---")
            for ex in examples:
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

    if phase == 'finetune':
        print("\n--- Esecuzione dello Split Stratificato per Fine-Tuning (90/10) ---")

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
    print(f"--- INIZIO FASE: {phase.upper()} ---\nUsing device: {device}")
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    train_dataloader, val_dataloader, tokenizer = get_ds(config, phase)

    model_config = {k: v for k, v in config.items() if k in ["d_model", "N", "h", "dropout", "d_ff", "seq_len"]}
    model = build_transformer(vocab_size=tokenizer.get_vocab_size(), multi_head=False, **model_config).to(device)
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
            is_new_phase = (phase == 'finetune' and 'pretrain' in preload_path)
            if is_new_phase:
                print("Inizio nuova fase (finetune): contatori resettati.")
                initial_epoch, global_step = 0, 0
            else:
                print("Continuazione di un training interrotto.")
                initial_epoch = state.get('epoch', -1) + 1
                global_step = state.get('global_step', 0)
                if 'optimizer_state_dict' in state: optimizer.load_state_dict(state['optimizer_state_dict'])
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

            decode_strategy_info = {}
            if phase == 'finetune':
                decode_strategy_info = {'strategy': 'greedy', 'repetition_penalty': 1.2}
            else:
                decode_strategy_info = {'strategy': 'greedy', 'repetition_penalty': 1.2}

            num_val_examples = config['num_validation_examples'] if phase == 'finetune' else -1

            run_validation(model, val_dataloader, tokenizer, config['seq_len'], device, global_step, writer,
                           num_val_examples, decode_strategy_info, phase)

            torch.save(
                {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                 'global_step': global_step}, f"{config['model_folder']}/{config['model_basename']}{epoch:02d}.pt")

    writer.close()
    print(f"--- FASE {phase.upper()} COMPLETATA ---")
    if phase == 'finetune':
        print("\n--- RUNNING FINAL EVALUATION WITH BEAM SEARCH ---")
        final_writer = SummaryWriter(config['experiment_name'] + "_final_beam_eval")
        beam_decode_info = {'strategy': 'beam', 'beam_size': 4, 'repetition_penalty': 1.2}
        run_validation(model, val_dataloader, tokenizer, config['seq_len'], device, global_step, final_writer, -1,
                       beam_decode_info, phase)
        final_writer.close()


if __name__ == '__main__':
    warnings.filterwarnings("ignore");
    os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
    parser = argparse.ArgumentParser(description='Train the NanoSocrates model in phases.');
    parser.add_argument('--phase', type=str, required=True, choices=['pretrain', 'finetune'])
    args = parser.parse_args()
    if args.phase == 'pretrain':
        config = get_pretrain_config()
    else:
        config = get_finetune_config()
    train_model(config, args.phase)