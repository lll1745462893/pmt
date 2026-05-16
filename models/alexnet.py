'''AlexNet in PyTorch.
Reference:
[1] Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton
    ImageNet Classification with Deep Convolutional Neural Networks. NIPS 2012.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class AlexNet(nn.Module):
    def __init__(self, num_classes=10, feat_scale=1):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        # For CIFAR-10/100 (32x32), the feature map size is 1x1 after pooling
        # For ImageNet (224x224), the feature map size is 6x6
        # We use adaptive pooling to handle different sizes
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x, with_latent=False):
        out = self.features(x)
        out = self.avgpool(out)
        latent = out.view(out.size(0), -1)
        final = self.classifier(latent)
        if with_latent:
            return final, latent
        return final


# Alias
alexnet = AlexNet


