# Multi-Head Classifier

Multi-task transformer classifier with 12 independent classification heads for adaptive/maladaptive behavior dimensions. Uses a shared HuggingFace encoder backbone.

## Setup

```bash
pip install torch transformers pandas tqdm
```

## Data

Place your CSV at the path specified in `config.py` (`all_timelines_merged.csv` by default). It must contain:
- A `text` column with the input text
- One column per head: `A_adaptive`, `A_maladaptive`, `B_O_adaptive`, `B_O_maladaptive`, `B_S_adaptive`, `B_S_maladaptive`, `C_O_adaptive`, `C_O_maladaptive`, `C_S_adaptive`, `C_S_maladaptive`, `D_adaptive`, `D_maladaptive`

## Training

```bash
python trainer.py --encoder mental/mental-bert-base-uncased
```

| Flag | Description | Default |
|------|-------------|---------|
| `--encoder` | HuggingFace model name (required) | — |
| `--tokenizer` | Tokenizer name (if different from encoder) | encoder value |
| `--epochs` | Number of training epochs | `3` |
| `--batch-size` | Batch size | `16` |
| `--lr` | Learning rate | `2e-5` |
| `--patience` | Early stopping patience (epochs without val loss improvement) | `3` |
| `--save-path` | Where to save final weights | `model.pt` |

The trainer saves two files: `model.pt` (final epoch) and `best_model.pt` (lowest validation loss). Use the best checkpoint for inference.

## Inference

```bash
python inference.py --post "sample text to classify" --model-path model.pt --encoder bert-base-uncased
```

| Flag | Description | Default |
|------|-------------|---------|
| `--post` | Input text to classify (required) | — |
| `--model-path` | Path to saved weights (required) | — |
| `--encoder` | HuggingFace model name (required) | — |
| `--tokenizer` | Tokenizer name (if different from encoder) | encoder value |
