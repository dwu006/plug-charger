import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))
        
    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pos_embedding[:, :seq_len, :]
        return self.dropout(x)

class TemporalTransformer(nn.Module):
    def __init__(self, input_dim=512, d_model=256, nhead=8, num_layers=4, dim_feedforward=1024, dropout=0.1, max_seq_len=100):
        super().__init__()
        if input_dim != d_model:
            self.input_proj = nn.Linear(input_dim, d_model)
        else:
            self.input_proj = nn.Identity()
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_seq_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x, mask=None):
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        if mask is not None:
            key_padding_mask = ~mask
        else:
            key_padding_mask = None
        output = self.transformer(x, src_key_padding_mask=key_padding_mask)
        return output

