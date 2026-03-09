import torch
import torch.nn as nn
import torchvision.models as models


class CNN(nn.Module):
    def __init__(self, outputDim=512, dropout=0.2):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        oldConv = resnet.conv1
        newConv = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            newConv.weight[:, :3] = oldConv.weight
            newConv.weight[:, 3] = oldConv.weight.mean(1)
        resnet.conv1 = newConv

        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, outputDim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return self.dropout(x)

    def freeze(self):
        for name, param in self.named_parameters():
            if any(f'backbone.{i}' in name for i in range(5)):
                param.requires_grad = False
            else:
                param.requires_grad = True

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True
