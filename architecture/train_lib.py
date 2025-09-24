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
import evaluate

from torch.utils.tensorboard import SummaryWriter

# Importa i moduli custom del progetto
from dataset_lib import NanoSocratesDataset, causal_mask
from model import build_transformer
from config import get_config


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


def run_validation(model, validation_ds, tokenizer, max_len, device, global_step, writer, num_examples_to_run):
    qualitative_examples = []
    NUM_QUALITATIVE_EXAMPLES = 5

    model.eval()

    count = len(validation_ds) if num_examples_to_run == -1 else num_examples_to_run

    rdf2text_preds, rdf2text_labels = [], []
    rdf_gen_tp, rdf_gen_fp, rdf_gen_fn = 0, 0, 0
    mlm_correct, mlm_total = 0, 0

    bleu_metric = evaluate.load("bleu")
    rouge_metric = evaluate.load("rouge")
    meteor_metric = evaluate.load("meteor")

    with torch.no_grad():
        batch_iterator = tqdm(validation_ds, desc="Validating", total=count)
        for i, batch in enumerate(batch_iterator):
            if i >= count:
                break

            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            model_out_tokens = greedy_decode(model, encoder_input, encoder_mask, tokenizer, max_len, device)
            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text_clean = tokenizer.decode(model_out_tokens.detach().cpu().numpy(), skip_special_tokens=True)
            model_out_text_raw = tokenizer.decode(model_out_tokens.detach().cpu().numpy(), skip_special_tokens=False)

            if i < NUM_QUALITATIVE_EXAMPLES:
                qualitative_examples.append({
                    "source": source_text,
                    "prediction": model_out_text_clean,
                    "ground_truth": target_text
                })

            if "<RDF2Text>" in source_text:
                rdf2text_preds.append(model_out_text_clean)
                rdf2text_labels.append([target_text])
            elif "<Text2RDF>" in source_text or "<CONTINUERDF>" in source_text:
                predicted_triples = parse_rdf_triples(model_out_text_raw)
                true_triples = parse_rdf_triples(target_text)
                rdf_gen_tp += len(predicted_triples.intersection(true_triples))
                rdf_gen_fp += len(predicted_triples.difference(true_triples))
                rdf_gen_fn += len(true_triples.difference(predicted_triples))
            elif "<MASK>" in source_text:
                if model_out_text_clean.strip() == target_text.strip():
                    mlm_correct += 1
                mlm_total += 1

    print("\n" + "=" * 80)
    if rdf2text_preds:
        bleu_score = bleu_metric.compute(predictions=rdf2text_preds, references=rdf2text_labels)
        rouge_score = rouge_metric.compute(predictions=rdf2text_preds, references=rdf2text_labels)
        meteor_score = meteor_metric.compute(predictions=rdf2text_preds, references=rdf2text_labels)
        writer.add_scalar("validation/bleu", bleu_score['bleu'], global_step)
        writer.add_scalar("validation/rougeL", rouge_score['rougeL'], global_step)
        writer.add_scalar("validation/meteor", meteor_score['meteor'], global_step)
        print("--- RDF2Text Metrics ---")
        print(f"BLEU:     {bleu_score['bleu']:.4f}")
        print(f"ROUGE-L:  {rouge_score['rougeL']:.4f}")
        print(f"METEOR:   {meteor_score['meteor']:.4f}")
    if (rdf_gen_tp + rdf_gen_fp > 0) or (rdf_gen_tp + rdf_gen_fn > 0):  # Condizione più sicura
        precision = rdf_gen_tp / (rdf_gen_tp + rdf_gen_fp) if (rdf_gen_tp + rdf_gen_fp) > 0 else 0
        recall = rdf_gen_tp / (rdf_gen_tp + rdf_gen_fn) if (rdf_gen_tp + rdf_gen_fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        writer.add_scalar("validation/rdf_precision", precision, global_step)
        writer.add_scalar("validation/rdf_recall", recall, global_step)
        writer.add_scalar("validation/rdf_f1", f1, global_step)
        print("--- Text2RDF / RDF Completion 2 Metrics ---")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
    if mlm_total > 0:
        accuracy = mlm_correct / mlm_total
        writer.add_scalar("validation/mlm_accuracy", accuracy, global_step)
        print("--- RDF Completion 1 (MLM) Metrics ---")
        print(f"Accuracy:  {accuracy:.4f}")
    print("=" * 80)

    current_epoch = global_step // len(train_dataloader) if len(train_dataloader) > 0 else 0
    print(f"--- Esempi Qualitativi (Fine Epoca {current_epoch}) ---")
    for idx, example in enumerate(qualitative_examples):
        print(f"\n----- Esempio {idx + 1} -----")
        print(f"INPUT      : {example['source']}")
        print(f"RIFERIMENTO: {example['ground_truth']}")
        print(f"PREDIZIONE : {example['prediction']}")
    print("\n" + "=" * 80 + "\n")

    table_markdown = "| Input | Riferimento | Predizione |\n|---|---|---|\n"
    for ex in qualitative_examples:
        # --- CORREZIONE PER SYNTAX WARNING ---
        clean_source = ex['source'].replace('|', '\\|')
        clean_truth = ex['ground_truth'].replace('|', '\\|')
        clean_pred = ex['prediction'].replace('|', '\\|')
        table_markdown += f"| {clean_source} | {clean_truth} | {clean_pred} |\n"
    writer.add_text('validation/qualitative_examples', table_markdown, global_step)

    writer.flush()
    model.train()

def get_ds(config):
    print(f"Caricamento tokenizer da: {config['tokenizer_file']}")
    tokenizer = Tokenizer.from_file(config['tokenizer_file'])

    full_dataset = NanoSocratesDataset(
        data_dir=config['data_dir'],
        tokenizer=tokenizer,
        seq_len=config['seq_len'],
        split='train'
    )

    train_ds_size = int(0.9 * len(full_dataset))
    val_ds_size = len(full_dataset) - train_ds_size
    train_ds, val_ds = random_split(full_dataset, [train_ds_size, val_ds_size])

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=4,
                                  pin_memory=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=False)

    return train_dataloader, val_dataloader, tokenizer



def get_model(config, vocab_size):
    return build_transformer(
        vocab_size=vocab_size,
        seq_len=config["seq_len"],
        d_model=config['d_model'],
        N=config['N'],
        h=config['h'],
        dropout=config['dropout'],
        d_ff=config['d_ff']
    )


def train_model(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available() and torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer = get_ds(config)
    model = get_model(config, tokenizer.get_vocab_size()).to(device)

    writer = SummaryWriter(config['experiment_name'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], eps=1e-9)

    initial_epoch = 0
    global_step = 0

    # Gestione del precaricamento di un modello
    if config['preload']:
        model_filename = config['preload']
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    total_steps = len(train_dataloader) * config['num_epochs']
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-7, last_epoch=global_step - 1)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id('<PAD>'), label_smoothing=0.1).to(device)

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
            loss = loss_fn(proj_output.view(-1, tokenizer.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}", "lr": f"{optimizer.param_groups[0]['lr']:.2e}"})

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


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)