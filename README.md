# DA6401 Assignment 3

Wandb Report link: [Click Here](https://wandb.ai/rohitrk06-indian-institute-of-technology-madras/da6401_assignment3_v1/reports/Assignment-3--VmlldzoxMjg3NTUyNQ?accessToken=omhhtcf3elx4m5gqwxtcgm7xf2ertyxwbzq7wmmdrylyi2e0cje4ilp4eah2is3c)

In this assignment we implements a **character-level sequence-to-sequence (Seq2Seq)** model for transliteration from Hindi to English using PyTorch Lightning. It supports both basic encoder-decoder and encoder-decoder with attention architectures.

---

In order to experiment with the model, we have used kaggle notebook for training and testing. The notebook used for that case are available in the this repository itself.
The notebook is named `rnn-with-attention.ipynb` and `rnn-without-attention.ipynb` for the attention and non-attention models respectively. 
If you want to run the code in your local machine, you can use the `train.py` file which is the main entry point for training the model.

The code is structured to allow easy switching between the two architectures. The attention mechanism is implemented in a separate module, and the main training script can toggle between using it or not.


## 📁 Project Structure
```
├── seq2seq_with_attention.py # Seq2Seq model with attention mechanism
├── seq2seq_without_attention.py # Basic Seq2Seq model without attention
├── utils.py # Dataset, DataModule, and utility functions
├── train.py # Training entry point
├── requirements.txt # Project dependencies
└── README.md # Project documentation (this file)
```
---

## 🔧 Components Overview

### `seq2seq_with_attention.py`

- Implements a full encoder-decoder model **with attention**.
- Modules:
  - **Encoder**: Character embedding + RNN/LSTM/GRU.
  - **AttentionModule**: Computes alignment scores and attention weights.
  - **Decoder**: Combines input embedding with attention context.
  - **Seq2Seq**: Wraps the training loop using `LightningModule`, manages teacher forcing, metrics, and prediction.

### `seq2seq_without_attention.py`

- Implements a standard Seq2Seq model **without attention**.
- Follows the same structure but lacks context vector computation and attention weights.

### `utils.py`

- **TransliterationDataset**: Prepares source and target sequences as tensors.
- **TrasnliterationDataModule** (typo: should be `TransliterationDataModule`): Manages data loading, vocabulary creation, batching, and padding.
- **decode_sequence()**: Converts output index tensors back to human-readable strings.

### `train.py`

- Main training script.
- Configurable via command-line arguments.
- Dynamically switches between attention and non-attention models.
- Uses `wandb` and PyTorch Lightning's `Trainer` for logging and training.
- Saves trained model to `.pkl` file.

---

## 📦 Installation

1. Clone the repository.
2. Install the dependencies:

```bash
pip install -r requirements.txt
```
3. Set up your environment for `wandb` if you haven't already:

```bash
pip install wandb
wandb login
```

## Dataset
This project expects the Dakshina dataset in the following structure:
```
dakshina_dataset_v1.0_hi/
└── lexicons/
    ├── hi.translit.sampled.train.tsv
    ├── hi.translit.sampled.dev.tsv
    └── hi.translit.sampled.test.tsv
```

## Usage

To train the model with attention, run:

```bash
python train.py --use_attention
```
To train the model without attention, run:

```bash
python train.py
```

Additonal Arguments can be passed to customize the training process. Here are the available arguments:

| Argument                  | Description                            | Default                             |
| ------------------------- | -------------------------------------- | ----------------------------------- |
| `--data_dir`              | Path to dataset folder                 | `dakshina_dataset_v1.0_hi/lexicons` |
| `--batch_size`            | Batch size                             | 128                                 |
| `--embedding_dimension`   | Character embedding dimension          | 512                                 |
| `--hidden_layer_size`     | Hidden size for RNN/LSTM/GRU layers    | 512                                 |
| `--num_layers`            | Number of encoder/decoder layers       | 3                                   |
| `--dropout`               | Dropout rate                           | 0.4                                 |
| `--cell_type`             | RNN cell type: `RNN`, `LSTM`, or `GRU` | `LSTM`                              |
| `--learning_rate`         | Learning rate                          | 0.0001                              |
| `--teacher_forcing_ratio` | Teacher forcing ratio                  | 0.5                                 |
| `--max_epochs`            | Training epochs                        | 10                                  |



