import torch
import torch.nn as nn
import torchvision.models as models

class CNN(nn.Module):
    def __init__(self, output_dim=512, pretrained=True, dropout=0.2):
        super().__init__()
        if pretrained:
            resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            resnet = models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        features = self.backbone(x)
        features = self.avgpool(features)
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        features = self.dropout(features)
        return features
    
    def freeze(self):
        for name, param in self.named_parameters():
            if any(f'backbone.{i}' in name for i in range(5)):
                param.requires_grad = False
            else:
                param.requires_grad = True
    
    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True
