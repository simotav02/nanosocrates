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
    model.eval()

    # Determina il numero di esempi da validare
    count = len(validation_ds) if num_examples_to_run == -1 else num_examples_to_run

    # Liste e contatori per ogni task
    rdf2text_preds, rdf2text_labels = [], []
    rdf_gen_tp, rdf_gen_fp, rdf_gen_fn = 0, 0, 0
    mlm_correct, mlm_total = 0, 0
    qualitative_examples = []
    NUM_QUALITATIVE_EXAMPLES = 5  # Numero di esempi da stampare a console

    # Carica le metriche dalla libreria evaluate
    bleu_metric = evaluate.load("bleu")
    rouge_metric = evaluate.load("rouge")
    meteor_metric = evaluate.load("meteor")

    with torch.no_grad():
        # Itera sul dataset di validazione con una barra di progresso
        batch_iterator = tqdm(validation_ds, desc="Validating", total=count)
        for i, batch in enumerate(batch_iterator):
            if i >= count:
                break

            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)

            # Esegui la decodifica greedy
            model_out_tokens = greedy_decode(model, encoder_input, encoder_mask, tokenizer, max_len, device)

            # Estrai i testi sorgente, target e predetto
            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text_clean = tokenizer.decode(model_out_tokens.detach().cpu().numpy(), skip_special_tokens=True)
            model_out_text_raw = tokenizer.decode(model_out_tokens.detach().cpu().numpy(), skip_special_tokens=False)

            # Salva i primi esempi per l'analisi qualitativa a console
            if len(qualitative_examples) < NUM_QUALITATIVE_EXAMPLES:
                qualitative_examples.append({
                    "source": source_text,
                    "prediction": model_out_text_clean,
                    "ground_truth": target_text
                })

            # Smista l'esempio nel task corretto in base al token di controllo
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
                mlm_total += 1
                if model_out_text_clean.strip() == target_text.strip():
                    mlm_correct += 1

    # --- CALCOLO E STAMPA DELLE METRICHE A CONSOLE ---
    print("\n" + "=" * 80)

    # Task: RDF-to-Text
    if rdf2text_preds:
        bleu_score = bleu_metric.compute(predictions=rdf2text_preds, references=rdf2text_labels)
        rouge_score = rouge_metric.compute(predictions=rdf2text_preds, references=rdf2text_labels)
        meteor_score = meteor_metric.compute(predictions=rdf2text_preds, references=rdf2text_labels)
        print("--- RDF2Text Metrics ---")
        print(f"BLEU:     {bleu_score['bleu']:.4f}")
        print(f"ROUGE-L:  {rouge_score['rougeL']:.4f}")
        print(f"METEOR:   {meteor_score['meteor']:.4f}\n")

    # Task: Text-to-RDF e RDF Completion 2
    if (rdf_gen_tp + rdf_gen_fp > 0) or (rdf_gen_tp + rdf_gen_fn > 0):
        precision = rdf_gen_tp / (rdf_gen_tp + rdf_gen_fp) if (rdf_gen_tp + rdf_gen_fp) > 0 else 0
        recall = rdf_gen_tp / (rdf_gen_tp + rdf_gen_fn) if (rdf_gen_tp + rdf_gen_fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        print("--- Text2RDF / RDF Completion 2 Metrics ---")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}\n")

    # Task: RDF Completion 1 (MLM)
    if mlm_total > 0:
        accuracy = mlm_correct / mlm_total
        print("--- RDF Completion 1 (MLM) Metrics ---")
        print(f"Accuracy:  {accuracy:.4f}\n")

    print("=" * 80)

    # --- STAMPA ESEMPI QUALITATIVI A CONSOLE ---
    print("--- Esempi Qualitativi ---")
    for idx, example in enumerate(qualitative_examples):
        print(f"\n----- Esempio {idx + 1} -----")
        print(f"INPUT      : {example['source']}")
        print(f"RIFERIMENTO: {example['ground_truth']}")
        print(f"PREDIZIONE : {example['prediction']}")
    print("\n" + "=" * 80 + "\n")

    # Riporta il modello in modalità training
    model.train()


def get_ds(config):
    tokenizer_path = Path(config['tokenizer_file'])
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer non trovato in '{tokenizer_path}'. Esegui prima tokenizer_lib.py.")

    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    # Carica i dati grezzi dai file
    source_path = os.path.join(config['data_dir'], "train.source")
    target_path = os.path.join(config['data_dir'], "train.target")

    with open(source_path, 'r', encoding='utf-8') as f:
        source_lines = f.readlines()
    with open(target_path, 'r', encoding='utf-8') as f:
        target_lines = f.readlines()

    # Crea una lista di coppie
    raw_ds = [{'source': src.strip(), 'target': tgt.strip()} for src, tgt in zip(source_lines, target_lines)]

    # Dividi i dati: 90% training, 10% validazione
    train_ds_size = int(0.9 * len(raw_ds))
    val_ds_size = len(raw_ds) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(raw_ds, [train_ds_size, val_ds_size])

    # Crea le istanze di NanoSocratesDataset usando i sottoinsiemi di dati
    train_ds = NanoSocratesDataset(train_ds_raw, tokenizer, config['seq_len'])
    val_ds = NanoSocratesDataset(val_ds_raw, tokenizer, config['seq_len'])

    # Stampa le lunghezze massime per un controllo (opzionale ma utile)
    max_len_src = 0
    max_len_tgt = 0
    for item in raw_ds:
        src_ids = tokenizer.encode(item['source']).ids
        tgt_ids = tokenizer.encode(item['target']).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=False)  # Batch size 1 per validazione

    # NOTA: Usiamo un solo tokenizer per source e target
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
    # Fallback per Mac M1/M2/M3
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