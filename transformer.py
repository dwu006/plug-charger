import torch
import torch.nn as nn

class PosEncoder(nn.Module):
    def __init__(self, dim, dropout=0.1, maxLen=100):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Parameter(torch.randn(1, maxLen, dim))

    def forward(self, x):
        x += self.embedding[:, :x.size(1), :]
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(self, inputDim=512, hiddenDim=256, nhead=8, numLayers=4, dropout=0.1, maxSeqLen=100):
        super().__init__()
        self.proj = nn.Linear(inputDim, hiddenDim)
        self.encoding = PosEncoder(hiddenDim, dropout, maxSeqLen)
        encoderLayer = nn.TransformerEncoderLayer(d_model=hiddenDim, nhead=nhead, dim_feedforward=hiddenDim*4, dropout=dropout, batch_first=True, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoderLayer, num_layers=numLayers)

    def forward(self, x):
        x = self.proj(x)
        x = self.encoding(x)
        return self.transformer(x)
