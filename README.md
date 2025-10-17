# NanoSocrates: Building a Very Small Semantic Language Model

This repository contains the official implementation for the **NanoSocrates** project, as part of the Deep Learning course at Politecnico di Bari.

The goal of this project is to build a unified, T5-style Transformer model from scratch, capable of understanding and translating between natural language text and structured RDF data. The model is trained and evaluated on four distinct tasks in the movie domain:

1.  **Text-to-RDF (Text2RDF)**: Generating RDF triples from text.
2.  **RDF-to-Text (RDF2Text)**: Generating text from RDF triples.
3.  **RDF Completion 1 (Masked Language Modeling)**: Predicting a masked component within a triple.
4.  **RDF Completion 2 (RDF Generation)**: Completing a given set of triples with new ones.

## Setup

### 1. Install Dependencies
First, install all required Python packages using the provided `requirements.txt` file. It is highly recommended to use a virtual environment.

`pip install -r requirements.txt`

### 2. Download NLTK Data
After installation, some evaluation metrics require NLTK data models. Run the following command in your terminal to download them. This only needs to be done once.

`python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"`

## Pipeline Execution

The scripts are designed to be run in a specific sequence to fetch, process, and prepare the data before training the model.

### Step 1: Data Acquisition

This script queries the public DBpedia SPARQL endpoint to collect raw data about films, including their abstracts from Wikipedia and associated RDF triples.

`python data_creation_2.py`

### Step 2: Data Preparation & Tokenization

This step prepares the raw data for both pre-training and fine-tuning.

**2.1 - Pre-training Corpus & Tokenizer**
These scripts create a large, plain-text corpus, train a custom BPE tokenizer on it, and then generate the T5-style masked dataset for the pre-training phase.

`python create_pretrain_corpus.py`
`python tokenizer_pretrain.py`
`python pretrain_dateset_T5.py`

**2.2 - Multi-Task Fine-tuning Dataset**
This script processes the JSON file to generate the final, balanced multi-task dataset (`Text2RDF`, `RDF2Text`, etc.) used for fine-tuning the model.

`python data_preprocessing_2.py`

### Step 3: Model Training

The `train_final.py` script handles all training phases. The desired phase must be specified using the `--phase` command-line argument.

**Note**: Between phases, you should update the corresponding configuration file (`config_pretrain.py`, etc.) to set the `preload` path to the last saved model checkpoint (`.pt` file).

`python train_final.py --phase pretrain`
`python train_final.py --phase decoder_tune`
`python train_final.py --phase full_finetune`

## Monitoring Training with TensorBoard

The training script saves logs to a directory specified in the config files (e.g., `runs/...`). To visualize training progress, such as the loss curves, you can use TensorBoard.

Run the following command from your project's main directory:

`tensorboard --logdir=runs`

Then, open the URL provided in your terminal (usually `http://localhost:6006`) in a web browser.

---

