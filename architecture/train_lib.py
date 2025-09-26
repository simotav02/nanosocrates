import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
# from torch.optim.lr_scheduler import CosineAnnealingLR
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


# scheduler con warm-up
def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


# Le funzioni greedy_decode, parse_rdf_triples, run_validation, get_ds, get_model
# rimangono invariate. Le ometto per brevità ma devono essere presenti nel tuo file.

def greedy_decode(model, source, source_mask, tokenizer, max_len, device):
    sot_idx = tokenizer.token_to_id('<SOT>')
    eot_idx = tokenizer.token_to_id('<EOT>')
    encoder_output = model.encode(source, source_mask)
    decoder_input = torch.empty(1, 1).fill_(sot_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break
        # La maschera del decoder deve essere ricreata ad ogni step con la nuova dimensione
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
    count = -1 if num_examples_to_run == -1 else num_examples_to_run

    # Liste per le metriche
    rdf2text_preds, rdf2text_labels = [], []
    rdf_gen_tp, rdf_gen_fp, rdf_gen_fn = 0, 0, 0
    mlm_correct, mlm_total = 0, 0
    qualitative_examples = []
    NUM_QUALITATIVE_EXAMPLES = 5

    # Caricamento metriche
    bleu_metric = evaluate.load("bleu")
    rouge_metric = evaluate.load("rouge")
    meteor_metric = evaluate.load("meteor")

    with torch.no_grad():
        for i, batch in enumerate(tqdm(validation_ds, desc="Validating")):
            if i == count:
                break
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)

            # Esegui la decodifica greedy per ottenere i token predetti
            model_out_tokens = greedy_decode(model, encoder_input, encoder_mask, tokenizer, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]

            # --- MODIFICA 1: Manteniamo entrambe le versioni della decodifica ---
            # Versione per valutazione di testo (es. BLEU, ROUGE)
            model_out_text_clean = tokenizer.decode(model_out_tokens.detach().cpu().numpy(), skip_special_tokens=True)
            # Versione per valutazione di RDF (che richiede i token speciali) e per il debug
            model_out_text_raw = tokenizer.decode(model_out_tokens.detach().cpu().numpy(), skip_special_tokens=False)

            # --- MODIFICA 2: Logica per selezionare l'output corretto da mostrare ---
            # Questa è la modifica cruciale per l'analisi qualitativa.
            display_prediction = ""
            if "<RDF2Text>" in source_text:
                # Per la generazione di testo, mostriamo il testo pulito.
                display_prediction = model_out_text_clean
            elif "<Text2RDF>" in source_text or "<CONTINUERDF>" in source_text:
                # Per la generazione di RDF, mostriamo l'output RAW con i token speciali.
                display_prediction = model_out_text_raw
            elif "<MASK>" in source_text:
                # Per MLM, l'output è una singola entità, quindi il testo pulito va bene.
                display_prediction = model_out_text_clean

            # Aggiungi agli esempi qualitativi la versione corretta della predizione
            if len(qualitative_examples) < NUM_QUALITATIVE_EXAMPLES:
                qualitative_examples.append(
                    {"source": source_text, "prediction": display_prediction, "ground_truth": target_text})

            # --- Il calcolo delle metriche rimane quasi invariato, ma ci assicuriamo che usi le variabili corrette ---
            if "<RDF2Text>" in source_text:
                # Per le metriche testuali, usiamo l'output pulito
                rdf2text_preds.append(model_out_text_clean)
                rdf2text_labels.append([target_text])
            elif "<Text2RDF>" in source_text or "<CONTINUERDF>" in source_text:
                # Per il parsing delle triple, usiamo l'output RAW
                predicted_triples = parse_rdf_triples(model_out_text_raw)
                true_triples = parse_rdf_triples(target_text)  # target_text ha già i token
                rdf_gen_tp += len(predicted_triples.intersection(true_triples))
                rdf_gen_fp += len(predicted_triples.difference(true_triples))
                rdf_gen_fn += len(true_triples.difference(predicted_triples))
            elif "<MASK>" in source_text:
                mlm_total += 1
                # Per l'accuracy di MLM, confrontiamo le stringhe pulite
                if model_out_text_clean.strip() == target_text.strip():
                    mlm_correct += 1


    print("\n" + "=" * 80)
    if rdf2text_preds:
        bleu_score = bleu_metric.compute(predictions=rdf2text_preds, references=rdf2text_labels)
        rouge_score = rouge_metric.compute(predictions=rdf2text_preds, references=rdf2text_labels)
        meteor_score = meteor_metric.compute(predictions=rdf2text_preds, references=rdf2text_labels)
        print("--- RDF2Text Metrics ---")
        print(f"BLEU: {bleu_score['bleu']:.4f}")
        print(f"ROUGE-L: {rouge_score['rougeL']:.4f}")
        print(f"METEOR: {meteor_score['meteor']:.4f}\n")
        writer.add_scalar('validation/bleu', bleu_score['bleu'], global_step)
        writer.add_scalar('validation/rougeL', rouge_score['rougeL'], global_step)
        writer.add_scalar('validation/meteor', meteor_score['meteor'], global_step)

    if (rdf_gen_tp + rdf_gen_fp > 0) or (rdf_gen_tp + rdf_gen_fn > 0):
        precision = rdf_gen_tp / (rdf_gen_tp + rdf_gen_fp) if (rdf_gen_tp + rdf_gen_fp) > 0 else 0
        recall = rdf_gen_tp / (rdf_gen_tp + rdf_gen_fn) if (rdf_gen_tp + rdf_gen_fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        print("--- Text2RDF / RDF Completion 2 Metrics ---")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}\n")
        writer.add_scalar('validation/rdf_f1', f1, global_step)
        writer.add_scalar('validation/rdf_precision', precision, global_step)
        writer.add_scalar('validation/rdf_recall', recall, global_step)

    if mlm_total > 0:
        accuracy = mlm_correct / mlm_total
        print("--- RDF Completion 1 (MLM) Metrics ---")
        print(f"Accuracy: {accuracy:.4f}\n")
        writer.add_scalar('validation/mlm_accuracy', accuracy, global_step)

    print("=" * 80)
    print("--- Esempi Qualitativi ---")
    for idx, example in enumerate(qualitative_examples):
        print(f"\n----- Esempio {idx + 1} -----")
        print(f"INPUT      : {example['source']}")
        print(f"RIFERIMENTO: {example['ground_truth']}")
        print(f"PREDIZIONE : '{example['prediction']}'")
    print("\n" + "=" * 80 + "\n")

    model.train()


def get_ds(config):
    tokenizer = Tokenizer.from_file(str(Path(config['tokenizer_file'])))
    raw_ds = []
    with open(os.path.join(config['data_dir'], "train.source"), 'r', encoding='utf-8') as f_src, \
            open(os.path.join(config['data_dir'], "train.target"), 'r', encoding='utf-8') as f_tgt:
        for src_line, tgt_line in zip(f_src, f_tgt):
            raw_ds.append({'source': src_line.strip(), 'target': tgt_line.strip()})

    train_ds_size = int(0.9 * len(raw_ds))
    val_ds_size = len(raw_ds) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(raw_ds, [train_ds_size, val_ds_size])

    train_ds = NanoSocratesDataset(train_ds_raw, tokenizer, config['seq_len'])
    val_ds = NanoSocratesDataset(val_ds_raw, tokenizer, config['seq_len'])

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=False)

    return train_dataloader, val_dataloader, tokenizer


def get_model(config, vocab_size):
    # Assicurati che il modello sia quello finale, con i moduli separati per src/tgt
    model = build_transformer(
        vocab_size=vocab_size,
        seq_len=config["seq_len"],
        d_model=config['d_model'],
        N=config['N'],
        h=config['h'],
        dropout=config['dropout'],
        d_ff=config['d_ff']
    )
    return model


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

    if config['preload']:
        model_filename = config['preload']
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
        print(f"Resuming training from epoch {initial_epoch}, global step {global_step}")

    # --- MODIFICA 3: Configurazione dello scheduler con warm-up lineare ---
    num_training_steps = len(train_dataloader) * config['num_epochs']
    num_warmup_steps = int(num_training_steps * 0.1)  # 10% di warm-up

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # Porta lo scheduler allo step corretto se si riprende il training
    if config['preload']:
        for _ in range(global_step):
            scheduler.step()

    # Per il training reale, riabilitiamo il label smoothing!
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id('<PAD>'), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")

        for batch in batch_iterator:
            optimizer.zero_grad(set_to_none=True)

            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)

            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)

            label = batch['label'].to(device)
            loss = loss_fn(proj_output.view(-1, tokenizer.get_vocab_size()), label.view(-1))

            current_lr = optimizer.param_groups[0]['lr']
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}", "lr": f"{current_lr:.2e}"})

            writer.add_scalar('train_loss', loss.item(), global_step)
            writer.add_scalar('learning_rate', current_lr, global_step)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Lo scheduler si aggiorna ad ogni STEP
            scheduler.step()

            global_step += 1

        # Esegui validazione e salva il modello alla fine di ogni epoca
        run_validation(model, val_dataloader, tokenizer, config['seq_len'], device, global_step, writer,
                       config['num_validation_examples'])

        model_filename = f"{config['model_folder']}/{config['model_basename']}{epoch:02d}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

    writer.close()


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)