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

from torch.utils.tensorboard import SummaryWriter

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


# --- Funzioni di Decodifica con la correzione della maschera ---

def greedy_decode(model, source, source_mask, tokenizer, max_len, device):
    sot_idx = tokenizer.token_to_id('<SOT>')
    eot_idx = tokenizer.token_to_id('<EOT>')
    pad_idx = tokenizer.token_to_id('<PAD>')
    encoder_output = model.encode(source, source_mask)
    decoder_input = torch.empty(1, 1).fill_(sot_idx).type_as(source).to(device)
    while decoder_input.size(1) < max_len:
        # CORREZIONE: Crea la maschera per l'input corrente del decoder
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
                completed_beams.append((candidate_seq, candidate_score));
                continue
            has_active_beams = True
            # CORREZIONE: Crea la maschera per la sequenza candidata
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
            completed_beams.extend(beams);
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
        # CORREZIONE: Crea la maschera per l'input corrente del decoder
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
                   decode_strategy='sampling'):
    # ... (Il resto della funzione run_validation è identico, gestisce già le diverse strategie)
    model.eval()
    count = -1 if num_examples_to_run == -1 else num_examples_to_run
    rdf2text_preds, rdf2text_labels = [], []
    rdf_gen_tp, rdf_gen_fp, rdf_gen_fn = 0, 0, 0
    mlm_correct, mlm_total = 0, 0
    qualitative_examples = []
    NUM_QUALITATIVE_EXAMPLES = 5
    BEAM_SIZE = 5
    bleu_metric = evaluate.load("bleu")
    rouge_metric = evaluate.load("rouge")
    meteor_metric = evaluate.load("meteor")

    with torch.no_grad():
        desc = f"Validating ({decode_strategy.capitalize()})"
        for i, batch in enumerate(tqdm(validation_ds, desc=desc)):
            if i == count: break
            encoder_input = batch["encoder_input"].to(device)
            encoder_padding_mask = batch["encoder_mask"].to(device)

            if decode_strategy == 'beam':
                model_out_tokens = beam_search_decode(model, BEAM_SIZE, encoder_input, encoder_padding_mask, tokenizer,
                                                      max_len, device)
            elif decode_strategy == 'sampling':
                model_out_tokens = top_k_sampling_decode(model, encoder_input, encoder_padding_mask, tokenizer, max_len,
                                                         device, k=50)
            else:
                model_out_tokens = greedy_decode(model, encoder_input, encoder_padding_mask, tokenizer, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text_clean = tokenizer.decode(model_out_tokens.detach().cpu().numpy(), skip_special_tokens=True)
            model_out_text_raw = tokenizer.decode(model_out_tokens.detach().cpu().numpy(), skip_special_tokens=False)

            if len(qualitative_examples) < NUM_QUALITATIVE_EXAMPLES:
                display_prediction = model_out_text_clean if "<RDF2Text>" in source_text or "<MLM>" in source_text else model_out_text_raw
                qualitative_examples.append(
                    {"source": source_text, "prediction": display_prediction, "ground_truth": target_text})

            if "<RDF2Text>" in source_text:
                rdf2text_preds.append(model_out_text_clean or ".")
                rdf2text_labels.append([target_text])
            elif "<Text2RDF>" in source_text or "<CONTINUERDF>" in source_text:
                predicted_triples = parse_rdf_triples(model_out_text_raw)
                true_triples = parse_rdf_triples(target_text)
                rdf_gen_tp += len(predicted_triples.intersection(true_triples))
                rdf_gen_fp += len(predicted_triples.difference(true_triples))
                rdf_gen_fn += len(true_triples.difference(predicted_triples))
            elif "<MLM>" in source_text:
                mlm_total += 1
                if model_out_text_clean.strip() == target_text.strip(): mlm_correct += 1

    print("\n" + "=" * 80)
    if rdf2text_preds:
        bleu_score = bleu_metric.compute(predictions=rdf2text_preds, references=rdf2text_labels)
        rouge_score = rouge_metric.compute(predictions=rdf2text_preds, references=rdf2text_labels)
        meteor_score = meteor_metric.compute(predictions=rdf2text_preds, references=rdf2text_labels)
        print("--- RDF2Text Metrics ---")
        print(f"BLEU: {bleu_score['bleu']:.4f}");
        print(f"ROUGE-L: {rouge_score['rougeL']:.4f}");
        print(f"METEOR: {meteor_score['meteor']:.4f}\n")
        if writer:
            writer.add_scalar(f'validation/{decode_strategy}/bleu', bleu_score['bleu'], global_step)
            writer.add_scalar(f'validation/{decode_strategy}/rougeL', rouge_score['rougeL'], global_step)
            writer.add_scalar(f'validation/{decode_strategy}/meteor', meteor_score['meteor'], global_step)

    if (rdf_gen_tp + rdf_gen_fp > 0) or (rdf_gen_tp + rdf_gen_fn > 0):
        precision = rdf_gen_tp / (rdf_gen_tp + rdf_gen_fp) if (rdf_gen_tp + rdf_gen_fp) > 0 else 0
        recall = rdf_gen_tp / (rdf_gen_tp + rdf_gen_fn) if (rdf_gen_tp + rdf_gen_fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        print("--- Text2RDF / RDF Completion 2 Metrics ---")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}\n")
        if writer:
            writer.add_scalar(f'validation/{decode_strategy}/rdf_f1', f1, global_step)
            writer.add_scalar(f'validation/{decode_strategy}/rdf_precision', precision, global_step)
            writer.add_scalar(f'validation/{decode_strategy}/rdf_recall', recall, global_step)

    if mlm_total > 0:
        accuracy = mlm_correct / mlm_total
        print("--- RDF Completion 1 (MLM) Metrics ---")
        print(f"Accuracy: {accuracy:.4f}\n")
        if writer: writer.add_scalar(f'validation/{decode_strategy}/mlm_accuracy', accuracy, global_step)

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
    source_path = os.path.join(config['data_dir'], "train.source")
    target_path = os.path.join(config['data_dir'], "train.target")
    if not os.path.exists(source_path) or not os.path.exists(target_path):
        raise FileNotFoundError(
            f"File di training non trovati in {config['data_dir']}. Assicurati di aver generato il dataset.")
    with open(source_path, 'r', encoding='utf-8') as f_src, open(target_path, 'r', encoding='utf-8') as f_tgt:
        raw_ds = [{'source': src_line.strip(), 'target': tgt_line.strip()} for src_line, tgt_line in zip(f_src, f_tgt)]
    train_ds_size = int(0.9 * len(raw_ds))
    val_ds_size = len(raw_ds) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(raw_ds, [train_ds_size, val_ds_size])
    train_ds = NanoSocratesDataset(train_ds_raw, tokenizer, config['seq_len'])
    val_ds = NanoSocratesDataset(val_ds_raw, tokenizer, config['seq_len'])
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=2,
                                  pin_memory=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=False)
    return train_dataloader, val_dataloader, tokenizer


def train_model(config):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"--- INIZIO TRAINING UNIFICATO ---")
    print(f"Using device: {device}")

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    train_dataloader, val_dataloader, tokenizer = get_ds(config)

    model = build_transformer(
        vocab_size=tokenizer.get_vocab_size(), seq_len=config["seq_len"], d_model=config['d_model'],
        N=config['N'], h=config['h'], dropout=config['dropout'], d_ff=config['d_ff']
    ).to(device)

    writer = SummaryWriter(config['experiment_name'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], eps=1e-9)

    initial_epoch, global_step = 0, 0
    if config['preload']:
        model_filename = config['preload']
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename, map_location=device)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
        print(f"Training resumes from epoch {initial_epoch}")

    num_training_steps = len(train_dataloader) * config['num_epochs']
    num_warmup_steps = int(num_training_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_training_steps, last_epoch=global_step - 1)

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id('<PAD>'), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
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

            current_lr = optimizer.param_groups[0]['lr']
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}", "lr": f"{current_lr:.2e}"})
            writer.add_scalar('train/step_loss', loss.item(), global_step)
            writer.add_scalar('train/learning_rate', current_lr, global_step)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1

        if (epoch + 1) % config.get('validate_every_n_epochs', 1) == 0:
            print(f"\n--- Running INTERMEDIATE validation for Epoch {epoch:02d} (fast with Sampling) ---")
            run_validation(
                model, val_dataloader, tokenizer, config['seq_len'], device,
                global_step, writer, config['num_validation_examples'],
                decode_strategy='sampling'
            )

            model_filename = f"{config['model_folder']}/{config['model_basename']}{epoch:02d}.pt"
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'global_step': global_step
            }, model_filename)

    writer.close()
    print("--- TRAINING COMPLETATO ---")

    print("\n--- RUNNING FINAL EVALUATION WITH BEAM SEARCH ---")
    final_writer = SummaryWriter(config['experiment_name'] + "_final_beam_eval")
    run_validation(
        model, val_dataloader, tokenizer, config['seq_len'], device,
        global_step, final_writer, config['num_validation_examples'],
        decode_strategy='beam'
    )
    final_writer.close()


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    # Sostituito get_config() con la nuova funzione unificata
    config = get_config()
    train_model(config)