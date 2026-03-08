import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, dim, dropout=0.1, max_len=100):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, dim))
        
    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pos_embedding[:, :seq_len, :]
        return self.dropout(x)

class TemporalTransformer(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, nhead=8, num_layers=4, dropout=0.1, max_seq_len=100):
        super().__init__()
        if input_dim != hidden_dim:
            self.proj = nn.Linear(input_dim, hidden_dim)
        else:
            self.proj = nn.Identity()
        self.pos_encoding = PositionalEncoding(hidden_dim, dropout, max_seq_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim*4, dropout=dropout, batch_first=True, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x, mask=None):
        x = self.proj(x)
        x = self.pos_encoding(x)
        if mask is not None:
            mask = ~mask
        output = self.transformer(x, src_key_padding_mask=mask)
        return output

