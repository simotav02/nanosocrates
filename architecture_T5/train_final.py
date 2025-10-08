# train_final.py (Script Unico con le modifiche richieste)

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
import argparse
import random
from collections import defaultdict, Counter

from torch.utils.tensorboard import SummaryWriter

from model_lib import build_transformer
from config_pretrain import get_pretrain_config, get_finetune_config
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


def beam_search_decode(model, beam_size, source, source_mask, tokenizer, max_len, device):
    sot_idx = tokenizer.token_to_id('<SOT>')
    eot_idx = tokenizer.token_to_id('<EOT>')
    pad_idx = tokenizer.token_to_id('<PAD>')
    encoder_output = model.encode(source, source_mask)
    initial_input = torch.empty(1, 1).fill_(sot_idx).type_as(source).to(device)
    beams = [(initial_input, 0.0)]
    completed_beams = []
    for _ in range(max_len - 1):
        new_beams, has_active_beams = [], False
        for candidate_seq, candidate_score in beams:
            if candidate_seq[0, -1].item() == eot_idx:
                completed_beams.append((candidate_seq, candidate_score))
                continue
            has_active_beams = True
            decoder_mask = (candidate_seq == pad_idx).to(device)
            out = model.decode(encoder_output, source_mask, candidate_seq, decoder_mask)
            log_probs = torch.log_softmax(model.project(out[:, -1]), dim=-1)
            topk_log_probs, topk_idx = torch.topk(log_probs, beam_size, dim=-1)
            for i in range(beam_size):
                token_idx = topk_idx[0, i].unsqueeze(0).unsqueeze(0)
                token_log_prob = topk_log_probs[0, i].item()
                new_seq = torch.cat([candidate_seq, token_idx], dim=1)
                new_beams.append((new_seq, candidate_score + token_log_prob))
        if not has_active_beams: break
        beams = sorted(new_beams, key=lambda x: x[1] / (x[0].size(1) ** 0.7), reverse=True)[:beam_size]
        if all(b[0][0, -1].item() == eot_idx for b in beams):
            completed_beams.extend(beams)
            break
    if not completed_beams: completed_beams = beams
    best_beam = sorted(completed_beams, key=lambda x: x[1] / (x[0].size(1) ** 0.7), reverse=True)[0]
    return best_beam[0].squeeze()


def top_k_sampling_decode(model, source, source_mask, tokenizer, max_len, device, k=50):
    sot_idx = tokenizer.token_to_id('<SOT>')
    eot_idx = tokenizer.token_to_id('<EOT>')
    pad_idx = tokenizer.token_to_id('<PAD>')
    encoder_output = model.encode(source, source_mask)
    decoder_input = torch.empty(1, 1).fill_(sot_idx).type_as(source).to(device)
    while decoder_input.size(1) < max_len:
        decoder_mask = (decoder_input == pad_idx).to(device)
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        logits = model.project(out[:, -1])
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
    if phase == 'pretrain':
        total_val_loss = 0
        total_correct_tokens = 0
        total_tokens = 0
        loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id('<PAD>')).to(device)
        with torch.no_grad():
            for batch in tqdm(validation_ds, desc="Validating Pretrain"):
                encoder_input = batch['encoder_input'].to(device)
                decoder_input = batch['decoder_input'].to(device)
                encoder_mask = batch['encoder_mask'].to(device)
                decoder_mask = batch['decoder_mask'].to(device)
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
            f"\n--- Pre-training Validation Metrics ---\nAverage Validation Loss: {avg_val_loss:.4f}\nToken-level Accuracy: {token_accuracy:.4f}\n")
        if writer:
            writer.add_scalar('validation/pretrain/loss', avg_val_loss, global_step)
            writer.add_scalar('validation/pretrain/token_accuracy', token_accuracy, global_step)
        print("--- Esempi Qualitativi di Denoising (Pre-training) ---")
        with torch.no_grad():
            first_batch = next(iter(validation_ds))
            for i in range(min(5, first_batch['encoder_input'].size(0))):
                encoder_input = first_batch["encoder_input"][i:i + 1].to(device)
                encoder_mask = first_batch["encoder_mask"][i:i + 1].to(device)
                model_out_tokens = greedy_decode(model, encoder_input, encoder_mask, tokenizer, max_len, device)
                source_text = first_batch["src_text"][i]
                target_text = first_batch["tgt_text"][i]
                model_out_text = tokenizer.decode(model_out_tokens.detach().cpu().numpy(), skip_special_tokens=False)
                print(
                    f"\n--- Esempio {i + 1} ---\nINPUT      : {source_text}\nRIFERIMENTO: {target_text}\nPREDIZIONE : {model_out_text}")
        print("\n" + "=" * 80 + "\n")
        model.train()
        return

    count = -1 if num_examples_to_run == -1 else num_examples_to_run
    rdf2text_preds, rdf2text_labels = [], []
    text2rdf_tp, text2rdf_fp, text2rdf_fn = 0, 0, 0
    continuerdf_tp, continuerdf_fp, continuerdf_fn = 0, 0, 0
    mlm_correct, mlm_total = 0, 0
    qualitative_examples = {}
    tasks_needed = {"Text2RDF", "RDF2Text", "MLM", "CONTINUERDF"}
    task_counter = Counter()
    sot_id = tokenizer.token_to_id('<SOT>')
    eot_id = tokenizer.token_to_id('<EOT>')

    with torch.no_grad():
        desc = f"Validating ({decode_strategy.capitalize()})"
        for i, batch in enumerate(tqdm(validation_ds, desc=desc)):
            if i == count and count != -1: break
            encoder_input = batch["encoder_input"].to(device)
            encoder_padding_mask = batch["encoder_mask"].to(device)
            if decode_strategy == 'beam':
                model_out_tokens = beam_search_decode(model, 5, encoder_input, encoder_padding_mask, tokenizer, max_len,
                                                      device)
            elif decode_strategy == 'sampling':
                model_out_tokens = top_k_sampling_decode(model, encoder_input, encoder_padding_mask, tokenizer, max_len,
                                                         device, k=50)
            else:
                model_out_tokens = greedy_decode(model, encoder_input, encoder_padding_mask, tokenizer, max_len, device)
            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            tokens_to_decode = model_out_tokens.detach().cpu().numpy()
            model_out_text_raw = tokenizer.decode(tokens_to_decode, skip_special_tokens=False)
            start_index = 1 if len(tokens_to_decode) > 0 and tokens_to_decode[0] == sot_id else 0
            eot_indices = [idx for idx, token_id in enumerate(tokens_to_decode) if token_id == eot_id]
            end_index = eot_indices[0] if eot_indices else len(tokens_to_decode)
            clean_tokens = tokens_to_decode[start_index:end_index]
            model_out_text_clean = tokenizer.decode(clean_tokens, skip_special_tokens=True).strip()
            current_task = None
            if "<RDF2Text>" in source_text:
                current_task = "RDF2Text"
            elif "<Text2RDF>" in source_text:
                current_task = "Text2RDF"
            elif "<CONTINUERDF>" in source_text:
                current_task = "CONTINUERDF"
            elif "<MLM>" in source_text:
                current_task = "MLM"

            if current_task and current_task in tasks_needed and current_task not in qualitative_examples:
                prediction_to_show = model_out_text_raw
                if current_task in ["RDF2Text", "MLM"]:
                    prediction_to_show = model_out_text_clean
                qualitative_examples[current_task] = {
                    "source": source_text, "prediction": prediction_to_show, "ground_truth": target_text
                }
            if current_task == "RDF2Text":
                task_counter['RDF2Text'] += 1
                rdf2text_preds.append(model_out_text_clean or ".")
                rdf2text_labels.append([target_text])
            elif current_task == "Text2RDF":
                task_counter['Text2RDF'] += 1
                predicted_triples = parse_rdf_triples(model_out_text_raw)
                true_triples = parse_rdf_triples(target_text)
                text2rdf_tp += len(predicted_triples.intersection(true_triples))
                text2rdf_fp += len(predicted_triples.difference(true_triples))
                text2rdf_fn += len(true_triples.difference(predicted_triples))
            elif current_task == "CONTINUERDF":
                task_counter['CONTINUERDF'] += 1
                predicted_triples = parse_rdf_triples(model_out_text_raw)
                true_triples = parse_rdf_triples(target_text)
                continuerdf_tp += len(predicted_triples.intersection(true_triples))
                continuerdf_fp += len(predicted_triples.difference(true_triples))
                continuerdf_fn += len(true_triples.difference(predicted_triples))
            elif current_task == "MLM":
                task_counter['MLM'] += 1
                mlm_total += 1
                if model_out_text_clean == target_text.strip():
                    mlm_correct += 1
            else:
                task_counter['Unknown'] += 1

    print("\n" + "=" * 80)
    print(f"Riepilogo task trovati nel validation set: {dict(task_counter)}")
    if rdf2text_preds:
        bleu = evaluate.load("bleu").compute(predictions=rdf2text_preds, references=rdf2text_labels)
        rouge = evaluate.load("rouge").compute(predictions=rdf2text_preds, references=rdf2text_labels)
        meteor = evaluate.load("meteor").compute(predictions=rdf2text_preds, references=rdf2text_labels)
        print(
            f"--- RDF2Text Metrics ---\nBLEU: {bleu['bleu']:.4f}, ROUGE-L: {rouge['rougeL']:.4f}, METEOR: {meteor['meteor']:.4f}\n")
        if writer: writer.add_scalar(f'validation/{decode_strategy}/rdf2text_bleu', bleu['bleu'], global_step)

    if (text2rdf_tp + text2rdf_fp > 0) or (text2rdf_tp + text2rdf_fn > 0):
        precision = text2rdf_tp / (text2rdf_tp + text2rdf_fp) if (text2rdf_tp + text2rdf_fp) > 0 else 0
        recall = text2rdf_tp / (text2rdf_tp + text2rdf_fn) if (text2rdf_tp + text2rdf_fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        print(f"--- Text2RDF Metrics ---\nPrecision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}\n")
        if writer: writer.add_scalar(f'validation/{decode_strategy}/text2rdf_f1', f1, global_step)

    if (continuerdf_tp + continuerdf_fp > 0) or (continuerdf_tp + continuerdf_fn > 0):
        precision = continuerdf_tp / (continuerdf_tp + continuerdf_fp) if (continuerdf_tp + continuerdf_fp) > 0 else 0
        recall = continuerdf_tp / (continuerdf_tp + continuerdf_fn) if (continuerdf_tp + continuerdf_fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        print(
            f"--- RDF Completion (CONTINUERDF) Metrics ---\nPrecision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}\n")
        if writer: writer.add_scalar(f'validation/{decode_strategy}/continuerdf_f1', f1, global_step)

    if mlm_total > 0:
        accuracy = mlm_correct / mlm_total
        print(f"--- RDF Completion (MLM) Metrics ---\nAccuracy: {accuracy:.4f}\n")
        if writer: writer.add_scalar(f'validation/{decode_strategy}/mlm_accuracy', accuracy, global_step)

    print("=" * 80)
    print("--- Esempi Qualitativi (Uno per Task) ---")
    for task_name in sorted(list(tasks_needed)):
        if task_name in qualitative_examples:
            ex = qualitative_examples[task_name]
            print(
                f"\n--- Esempio Task: {task_name} ---\nINPUT      : {ex['source']}\nRIFERIMENTO: {ex['ground_truth']}\nPREDIZIONE : '{ex['prediction']}'")
        else:
            print(f"\n--- Esempio Task: {task_name} ---\nNessun esempio trovato in questo batch di validazione.")
    print("\n" + "=" * 80 + "\n")
    model.train()


def get_ds(config, phase: str):
    tokenizer = Tokenizer.from_file(str(Path(config['tokenizer_file'])))
    raw_ds = []
    source_path = os.path.join(config['data_dir'], "train.source")
    target_path = os.path.join(config['data_dir'], "train.target")
    if not os.path.exists(source_path): raise FileNotFoundError(
        f"File di training non trovati in {config['data_dir']}.")
    with open(source_path, 'r', encoding='utf-8') as f_src, open(target_path, 'r', encoding='utf-8') as f_tgt:
        raw_ds = [{'source': s.strip(), 'target': t.strip()} for s, t in zip(f_src, f_tgt)]

    if phase == 'finetune':
        print("\n--- Esecuzione dello Split Stratificato per Fine-Tuning (90/10) ---")

        def get_task_category(source_text):
            if "<RDF2Text>" in source_text: return "RDF2Text"
            if "<Text2RDF>" in source_text: return "Text2RDF"
            if "<CONTINUERDF>" in source_text: return "CONTINUERDF"
            if "<MLM>" in source_text: return "MLM"
            return "Unknown"

        grouped_ds = defaultdict(list)
        for item in raw_ds: grouped_ds[get_task_category(item['source'])].append(item)
        train_raw, val_raw = [], []
        for category, items in sorted(grouped_ds.items()):
            random.shuffle(items)
            split_point = int(0.9 * len(items))
            train_raw.extend(items[:split_point])
            val_raw.extend(items[split_point:])
            print(f"Categoria '{category}': {len(items[:split_point])} train / {len(items[split_point:])} val")
    elif phase == 'pretrain':
        print("\n--- Esecuzione dello Split Casuale Semplice per Pre-Training (90/10) ---")
        random.shuffle(raw_ds)
        split_point = int(0.9 * len(raw_ds))
        train_raw = raw_ds[:split_point]
        val_raw = raw_ds[split_point:]
    else:
        raise ValueError(f"Fase '{phase}' non riconosciuta. Usare 'pretrain' o 'finetune'.")

    random.shuffle(train_raw)
    random.shuffle(val_raw)
    print(f"Totale: {len(train_raw)} esempi di training, {len(val_raw)} esempi di validazione.\n")
    train_ds = NanoSocratesDataset(train_raw, tokenizer, config['seq_len'])
    val_ds = NanoSocratesDataset(val_raw, tokenizer, config['seq_len'])
    train_dl = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
    val_batch_size = config['batch_size'] if phase == 'pretrain' else 1
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
    print(f"Ottimizzatore: AdamW con lr={config['lr']} e weight_decay=0.01")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], eps=1e-9, weight_decay=0.01)

    initial_epoch, global_step = 0, 0
    if config.get('preload'):
        print(f"Preloading model {config['preload']}")
        state = torch.load(config['preload'], map_location=device)
        model.load_state_dict(state['model_state_dict'])
        if phase == 'finetune':
            print("Fase di Fine-tuning: i contatori di epoca e step vengono resettati.")
        else:
            initial_epoch = state.get('epoch', -1) + 1
            optimizer.load_state_dict(state['optimizer_state_dict'])
            global_step = state.get('global_step', 0)
        print(f"Il training parte dall'epoca {initial_epoch}")

    # --- MODIFICA DELLO SCHEDULER ---
    print(f"Scheduler: LinearLR con decadimento lineare per {config['num_epochs']} epoche.")
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1,
                                                  total_iters=config['num_epochs'])

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id('<PAD>'),
                                  label_smoothing=config['loss_label_smoothing']).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        total_loss = 0.0
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:
            optimizer.zero_grad(set_to_none=True)
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)
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
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar(f'learning_rate/{phase}', current_lr, epoch)
        print(f"Learning rate per la prossima epoca: {current_lr:.2e}")

        if (epoch + 1) % config.get('validate_every_n_epochs', 1) == 0:
            print(f"\n--- Running validation for Epoch {epoch:02d} ---")
            run_validation(model, val_dataloader, tokenizer, config['seq_len'], device, global_step, writer,
                           config['num_validation_examples'], 'sampling', phase)
            torch.save(
                {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                 'global_step': global_step},
                f"{config['model_folder']}/{config['model_basename']}{epoch:02d}.pt")

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
    parser.add_argument('--phase', type=str, required=True, choices=['pretrain', 'finetune'])
    args = parser.parse_args()
    config = get_pretrain_config() if args.phase == 'pretrain' else get_finetune_config()
    train_model(config, args.phase)