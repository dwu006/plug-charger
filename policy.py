import torch
import torch.nn as nn
from cnn import CNN
from transformer import TemporalTransformer

class Policy(nn.Module):
    def __init__(self, action_dim=6, freeze_cnn=True): # output 6D (5joints + gripper)
        super().__init__()
        self.cnn = CNN(output_dim=512, pretrained=True)
        if freeze_cnn:
            self.cnn.freeze()
        self.transformer = TemporalTransformer(input_dim=512, hidden_dim=256, nhead=8, num_layers=4, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, action_dim))
        
    def forward(self, images, mask=None):
        batch, seq_len = images.shape[:2]
        images_flat = images.view(-1, 3, 224, 224)
        features = self.cnn(images_flat)
        features = features.view(batch, seq_len, 512)
        out = self.transformer(features, mask=mask)
        actions = self.head(out)
        return actions
    
    def predict(self, image):
        self.eval()
        with torch.no_grad():
            if image.dim() == 4:
                image = image.unsqueeze(1)
            actions = self.forward(image)
            return actions.squeeze(1)