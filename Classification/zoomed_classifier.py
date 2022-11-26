# coding: utf-8

# Imports

import torch
import torch.nn as nn
import torch.nn.functional as F


# Model

class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, act=True):
        padding = (kernel_size - 1) // 2
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels)
        ]
        if act: layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)


class BasicResidual(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels, act=False)
        )


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.shortcut = self.get_shortcut(in_channels, out_channels)
        self.residual = BasicResidual(in_channels, out_channels)
        self.act = nn.ReLU(inplace=True)
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        out = self.shortcut(x) + self.gamma * self.residual(x)
        return self.act(out)
    
    def get_shortcut(self, in_channels, out_channels):
        if in_channels != out_channels:
            shortcut = ConvBlock(in_channels, out_channels, 1, act=False)
        else:
            shortcut = nn.Identity()
        return shortcut


class ResidualStack(nn.Sequential):
    def __init__(self, in_channels, repetitions, strides):
        layers = []
        out_channels = in_channels
        for rep, stride in zip(repetitions, strides):
            if stride > 1:
                layers.append(nn.MaxPool2d(stride))
            for _ in range(rep):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
            out_channels = out_channels * 2
        super().__init__(*layers)


class Stem(nn.Sequential):
    def __init__(self, channel_list, stride):
        layers = [ConvBlock(*channel_list[:2], stride=stride)]
        for in_channels, out_channels in zip(channel_list[1:], channel_list[2:]):
            layers.append(ConvBlock(in_channels, out_channels))
        super().__init__(*layers)


class Head(nn.Sequential):
    def __init__(self, in_channels, classes, p_drop=0.):
        super().__init__(
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.Dropout(p_drop),
            nn.Linear(in_channels, classes)
        )


class ResNet(nn.Sequential):
    def __init__(self, classes, repetitions, strides=None, p_drop=0.):
        if strides is None: strides = [2] * (len(repetitions) + 1)
        super().__init__(
            Stem([3, 32, 32, 64], strides[0]),
            ResidualStack(64, repetitions, strides[1:]),
            Head(64 * 2**(len(repetitions) - 1), classes, p_drop)
        )


# Functions 

def create_classification_model(weights_path, device):
    model = ResNet(1, [2, 2, 2, 2], strides=[2, 2, 2, 2, 2], p_drop=0.3)
    model.to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval();
    return model


def image_to_tensor(img):
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img.float().div(255)


def classify_image(image, model, device, threshold=0.5):
#     image = image_to_tensor(image)
    output = model(image.to(device).unsqueeze(0))
    prob = torch.sigmoid(output).item()
    pred_label = prob > threshold
    return pred_label
