# inference.py

import argparse
import torch
from transformers import AutoTokenizer
from model import MultiHeadClassifier
from config import HEADS

def predict(post, model_path, encoder_name, tokenizer_name, device="cpu"):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = MultiHeadClassifier(encoder_name)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    inputs = tokenizer(post, truncation=True, max_length=256, padding="max_length", return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        predictions = [logit.argmax(dim=-1).item() for logit in logits]
    result = {head: pred for (head, _), pred in zip(HEADS, predictions)}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with multi-head classifier")
    parser.add_argument("--post", type=str, required=True, help="Input text to classify")
    parser.add_argument("--model-path", type=str, required=True, help="Path to saved model weights")
    parser.add_argument("--encoder", type=str, required=True, help="HuggingFace encoder model name")
    parser.add_argument("--tokenizer", type=str, default=None, help="HuggingFace tokenizer name (defaults to encoder)")
    args = parser.parse_args()

    tokenizer = args.tokenizer or args.encoder
    result = predict(args.post, args.model_path, args.encoder, tokenizer)
    for head, pred in result.items():
        print(f"{head}: {pred}")
