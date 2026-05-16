"""SENet for CIFAR-style inputs in PyTorch."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        self.fc1 = nn.Conv2d(planes, max(planes // 16, 1), kernel_size=1)
        self.fc2 = nn.Conv2d(max(planes // 16, 1), planes, kernel_size=1)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        w = F.adaptive_avg_pool2d(out, 1)
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        out = out * w

        out += self.shortcut(x)
        return F.relu(out)


class PreActBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.shortcut = None
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)

        self.fc1 = nn.Conv2d(planes, max(planes // 16, 1), kernel_size=1)
        self.fc2 = nn.Conv2d(max(planes // 16, 1), planes, kernel_size=1)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if self.shortcut is not None else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))

        w = F.adaptive_avg_pool2d(out, 1)
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        out = out * w

        out += shortcut
        return F.relu(out)


class SENet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for cur_stride in strides:
            layers.append(block(self.in_planes, planes, cur_stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x, with_latent=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1)
        pre_out = out.view(out.size(0), -1)
        final = self.linear(pre_out)
        if with_latent:
            return final, pre_out
        return final

    def custom_forward(self, x):
        activation = {}
        out = F.relu(self.bn1(self.conv1(x)))
        activation['pre-extract'] = out
        out = self.layer1(out)
        activation['layer1'] = out
        out = self.layer2(out)
        activation['layer2'] = out
        out = self.layer3(out)
        activation['layer3'] = out
        out = self.layer4(out)
        activation['layer4'] = out
        out = F.adaptive_avg_pool2d(out, 1)
        pre_out = out.view(out.size(0), -1)
        activation['post-extract'] = pre_out
        final = self.linear(pre_out)
        return final, activation


def SENet18(num_classes=10, **kwargs):
    return SENet(PreActBlock, [2, 2, 2, 2], num_classes=num_classes, **kwargs)


senet18 = SENet18
