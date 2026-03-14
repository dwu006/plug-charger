import torch
import torch.nn as nn
from cnn import CNN
from .transformer import Transformer

class Policy(nn.Module):
    def __init__(self, actionDim=6, stateDim=6, chunkSize=4):
        super().__init__()
        self.actionDim = actionDim
        self.chunkSize = chunkSize
        self.cnn = CNN(outputDim=512)
        self.cnn.freeze()
        self.transformer = Transformer(inputDim=512 + stateDim, hiddenDim=256, nhead=8, numLayers=4, dropout=0.1)
        self.head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, chunkSize * actionDim)
        )

    def forward(self, x, states):
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x = self.cnn(x.view(B * T, C, H, W)).view(B, T, -1)
        x = torch.cat([x, states], dim=-1)
        out = self.transformer(x)
        return self.head(out[:, -1, :]).view(-1, self.chunkSize, self.actionDim)

    def predict(self, images, states):
        self.eval()
        with torch.no_grad():
            T, C, H, W = images.shape
            features = self.cnn(images).unsqueeze(0)
            return self.forward(features, states.unsqueeze(0)).squeeze(0)
