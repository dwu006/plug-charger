import torch
import torch.nn as nn
from cnn import CNN
from transformer import Transformer


class Policy(nn.Module):
    def __init__(self, actionDim=6):
        super().__init__()
        self.cnn = CNN(outputDim=512)
        self.cnn.freeze()
        self.transformer = Transformer(inputDim=512, hiddenDim=256, nhead=8, numLayers=4, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, actionDim))

    def forward(self, images):
        features = self.cnn(images)            
        features = features.unsqueeze(1)      
        out = self.transformer(features)    
        out = out.squeeze(1)
        return self.head(out)    

    def predict(self, image):
        self.eval()
        with torch.no_grad():
            image = image.unsqueeze(0)
            return self.forward(image).squeeze(0)
