# model.py

import torch
import torch.nn as nn
from transformers import AutoModel
from config import HEADS

class MultiHeadClassifier(nn.Module):
    def __init__(self, encoder_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden_size = self.encoder.config.hidden_size
        self.heads = nn.ModuleList([
            nn.Linear(hidden_size, num_classes) for _, num_classes in HEADS
        ])

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]    # CLS token (batch, hidden)
        logits = [head(pooled) for head in self.heads] # List of (batch, num_classes)
        return logits