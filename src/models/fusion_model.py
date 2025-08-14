import torch
import torch.nn as nn
from src.utils.logging import lprint, LoggingLevels as ll

class TransformerFusionModel(nn.Module):
    def __init__(self, teacher_output_sizes, hidden_dim=128, nhead=5, num_layers=2, dropout=0.3, batch_norm=True):
        super(TransformerFusionModel, self).__init__()
        self.input_dim = sum(teacher_output_sizes)  # 5
        self.hidden_dim = hidden_dim
        self.nhead = nhead
        
        if hidden_dim % nhead != 0:
            lprint(ll.WARN, f"hidden_dim {hidden_dim} not divisible by nhead {nhead}. Adjusting nhead.")
            self.nhead = min([i for i in range(1, hidden_dim + 1) if hidden_dim % i == 0], key=lambda x: abs(x - nhead))
        
        self.input_projection = nn.Linear(self.input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=self.nhead,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(hidden_dim, len(teacher_output_sizes))
        self.batch_norm = nn.BatchNorm1d(hidden_dim) if batch_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, teacher_outputs, mask=None):
        try:
            x = torch.cat(teacher_outputs, dim=-1)  # [batch_size, 5]
            x = self.input_projection(x)  # [batch_size, hidden_dim]
            x = self.batch_norm(x)
            x = self.dropout(x)
            x = x.unsqueeze(1)  # [batch_size, 1, hidden_dim]
            x = self.transformer(x)  # [batch_size, 1, hidden_dim]
            x = x.squeeze(1)  # [batch_size, hidden_dim]
            return self.output_layer(x)  # [batch_size, 5]
        except Exception as e:
            lprint(ll.ERROR, f"Error in TransformerFusionModel forward: {str(e)}")
            raise