"""
MindSense AI - PyTorch Hybrid Classifier Model Definition
============================================================
Architecture: DistilBERT + BiLSTM + Token Attention
Matches research paper specification.

Author: MindSense AI Team
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 128


class AttentionLayer(nn.Module):
    """Token-level multiplicative attention layer over BiLSTM outputs."""

    def __init__(self, hidden_dim: int):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_dim * 2, 1)

    def forward(self, lstm_output: torch.Tensor):
        # lstm_output: (batch_size, seq_len, hidden_dim * 2)
        attn_weights = torch.softmax(self.attention(lstm_output), dim=1)
        context_vector = torch.sum(attn_weights * lstm_output, dim=1)
        return context_vector, attn_weights


class HybridMentalHealthModel(nn.Module):
    """
    Hybrid Transformer-Sequential model combining:
      1. DistilBERT contextual embeddings (768-dim)
      2. Bidirectional LSTM for sequence dynamics (128 hidden)
      3. Attention Layer for key token re-weighting
      4. Dropout (0.3) + Linear Classifier
    """

    def __init__(self, num_classes: int = 7):
        super(HybridMentalHealthModel, self).__init__()
        self.bert = AutoModel.from_pretrained(MODEL_NAME)
        self.bilstm = nn.LSTM(
            input_size=768,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.attention = AttentionLayer(128)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_output.last_hidden_state
        lstm_output, _ = self.bilstm(sequence_output)
        attention_output, attention_weights = self.attention(lstm_output)
        output = self.dropout(attention_output)
        logits = self.fc(output)
        return logits, attention_weights
