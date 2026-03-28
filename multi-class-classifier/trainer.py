# trainer.py

import argparse
import os
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split

# Must match evaluate.ipynb and test_train_info.ipynb (same generator + lengths → identical train/val indices).
SPLIT_SEED = 42
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from dataset import MultiHeadDataset
from model import MultiHeadClassifier
from config import CSV_FILE, HEADS


def evaluate(model, val_dl, criterion, device):
    model.eval()
    val_loss = 0.
    correct, total = [0]*len(HEADS), [0]*len(HEADS)
    with torch.no_grad():
        for batch in val_dl:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids, attention_mask)
            val_loss += sum(criterion(logit, labels[:, i]) for i, logit in enumerate(logits)).item() / len(logits)
            for i, logit in enumerate(logits):
                preds = logit.argmax(dim=-1)
                correct[i] += (preds == labels[:, i]).sum().item()
                total[i] += len(labels)
    avg_val_loss = val_loss / len(val_dl)
    accuracies = {head: correct[i] / total[i] for i, (head, _) in enumerate(HEADS)}
    return avg_val_loss, accuracies


def train(
    encoder_name,
    tokenizer_name,
    num_epochs=20,
    batch_size=16,
    lr=2e-5,
    validation_split=0.2,
    save_path="model.pt",
    patience=3,
    device="cuda" if torch.cuda.is_available() else "mps"
):

    dataset = MultiHeadDataset(CSV_FILE, tokenizer_name)
    val_size = int(len(dataset) * validation_split)
    train_size = len(dataset) - val_size
    split_generator = torch.Generator().manual_seed(SPLIT_SEED)
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size], generator=split_generator
    )
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)
    model = MultiHeadClassifier(encoder_name).to(device)

    optimizer = AdamW(model.parameters(), lr=lr)
    num_steps = len(train_dl) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_steps)

    criterion = torch.nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    checkpoint_dir = os.path.dirname(save_path) or "."
    best_path = os.path.join(checkpoint_dir, "best_" + os.path.basename(save_path))

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.
        for batch in tqdm(train_dl, desc=f"Epoch {epoch+1}/{num_epochs}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = sum(criterion(logit, labels[:, i]) for i, logit in enumerate(logits)) / len(logits)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_dl)
        print(f"Epoch {epoch+1} train loss: {avg_loss:.4f}")

        avg_val_loss, accuracies = evaluate(model, val_dl, criterion, device)
        print(f"Epoch {epoch+1} val loss:   {avg_val_loss:.4f}")
        for head, acc in accuracies.items():
            print(f"  {head} acc: {acc:.3f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_path)
            print(f"  Best model saved to {best_path}")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement}/{patience} epochs")
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered after epoch {epoch+1}")
                break

    torch.save(model.state_dict(), save_path)
    print(f"Final model saved to {save_path}")
    print(f"Best model (by val loss) at {best_path}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train multi-head classifier")
    parser.add_argument("--encoder", type=str, required=True, help="HuggingFace encoder model name")
    parser.add_argument("--tokenizer", type=str, default=None, help="HuggingFace tokenizer name (defaults to encoder)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--save-path", type=str, default="model.pt")
    args = parser.parse_args()

    tokenizer = args.tokenizer or args.encoder
    train(
        args.encoder, tokenizer,
        num_epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, save_path=args.save_path, patience=args.patience,
    )
