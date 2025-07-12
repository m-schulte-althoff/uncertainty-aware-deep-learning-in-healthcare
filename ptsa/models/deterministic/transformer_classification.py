import torch
from torch import nn
import torch.nn.functional as F
import math
from typing import Optional

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_length: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=0.0)

        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_seq_length, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerIHM(nn.Module):
    def __init__(
        self, 
        input_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        super(TransformerIHM, self).__init__()
        
        self.input_norm = nn.LayerNorm(input_size)
        self.input_projection = nn.Linear(input_size, d_model)
        torch.nn.init.xavier_uniform_(self.input_projection.weight, gain=0.1)
        
        self.input_dropout = nn.Dropout(p=dropout)
        
        self.pos_encoder = PositionalEncoding(d_model, dropout=0.0)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
            activation=activation,
            batch_first=True,
            norm_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.final_norm = nn.LayerNorm(d_model)
        self.final_dropout = nn.Dropout(p=dropout)
        
        self.fc_logit = nn.Linear(d_model, 1)
        
        torch.nn.init.xavier_uniform_(self.fc_logit.weight, gain=0.1)
        
        self.d_model = d_model

    def forward(
        self, 
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.input_norm(src)
        x = self.input_projection(x) * math.sqrt(self.d_model)
        # x = self.input_dropout(x)
        
        x = self.pos_encoder(x)
        
        x = self.transformer_encoder(
            x,
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask
        )
        
        x = x[:, -1, :]
        
        x = self.final_norm(x)
        # x = self.final_dropout(x)
        
        logits = self.fc_logit(x).squeeze(-1)
        
        proba = torch.clamp(torch.sigmoid(logits), min=1e-6, max=1-1e-6)
        
        return proba
