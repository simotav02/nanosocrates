import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tokenizers import Tokenizer
from pathlib import Path
from tqdm import tqdm
import warnings

# Importa i tuoi moduli custom
from architecture.dataset_lib import NanoSocratesDataset
from architecture.model import build_transformer
from architecture.config import get_config



def get_ds(config):
    print(f"Caricamento tokenizer da: {config['tokenizer_file']}")
    tokenizer = Tokenizer.from_file(config['tokenizer_file'])

    # Creazione del dataset usando il tokenizer caricato
    dataset = NanoSocratesDataset(config['corpus_file'], tokenizer, config['seq_len'])

    # Split in train e validation set
    train_ds_size = int(0.9 * len(dataset))
    val_ds_size = len(dataset) - train_ds_size
    train_ds, val_ds = random_split(dataset, [train_ds_size, val_ds_size])

    # Creazione dei DataLoader
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)  # Batch size 1 per la validazione è comune

    return train_dataloader, val_dataloader, tokenizer


def get_model(config, vocab_size):
    model = build_transformer(
        vocab_size,
        config["seq_len"],
        d_model=config['d_model'],
        N=config['N'],
        h=config['h'],
        dropout=config['dropout'],
        d_ff=config['d_ff']
    )
    return model


def train_model(config):
    # Selezione del device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using device: CUDA (NVIDIA GPU)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: MPS (Apple GPU)")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer = get_ds(config)

    vocab_size = tokenizer.get_vocab_size()
    print(f"Dimensione del vocabolario: {vocab_size}")

    model = get_model(config, vocab_size).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], eps=1e-9, weight_decay=0.01)
    initial_epoch = 0

    pad_token_id = tokenizer.token_to_id('<PAD>')
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id, label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()  # Utile su GPU
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")

        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)

            # Forward pass
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)

            label = batch['label'].to(device)

            # Calcolo della loss
            loss = loss_fn(proj_output.view(-1, vocab_size), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # TODO: Aggiungere un ciclo di validazione qui per monitorare le performance

        # Salvataggio del modello alla fine di ogni epoca
        model_filename = f"{config['model_folder']}/{config['model_basename']}{epoch:02d}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, model_filename)
        print(f"Modello salvato in {model_filename}")


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)