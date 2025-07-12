import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super(GRU, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.fc_log_var = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)

        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        h0: Tensor = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        mask = torch.ones_like(h0).bernoulli_(1 - self.dropout).to(x.device) / (1 - self.dropout)
        h0 = h0 * mask


        output, hidden = self.gru(x, h0)
        last_hidden = F.dropout(hidden[-1], p=self.dropout, training=self.training)
        
        proba = self.sigmoid(self.fc(last_hidden)).squeeze(-1)
        log_var = self.fc_log_var(last_hidden).squeeze(-1)
        
        return proba, log_var

            
    def predict_with_uncertainty(self, x, num_samples=100):
        self.train()
        
        predictions = []
        log_variances = []
        
        for _ in range(num_samples):
            with torch.no_grad():
                mean, log_var = self(x)
                predictions.append(mean)
                log_variances.append(log_var)

        predictions = torch.stack(predictions)
        mean = predictions.mean(dim=0)

        model_variance = predictions.var(dim=0)
        aleatoric_variance = torch.exp(torch.stack(log_variances).mean(dim=0))
        total_variance = model_variance + aleatoric_variance

        return mean, total_variance

