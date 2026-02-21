import torch
import torch.nn as nn
import torchvision.models as models

class CNN(nn.Module): # resnet18 backbone
    def __init__(self, output_dim=512, pretrained=True):
        super().__init__()

        if pretrained:
            resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            resnet = models.resnet18(weights=None)
        
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, output_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        features = self.backbone(x)
        features = self.avgpool(features)
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        features = self.dropout(features)
        return features
