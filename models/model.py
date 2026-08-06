import torchvision
from torch import nn
from torchvision.models import ResNet50_Weights
from config import config


def get_net():
    # PyTorch 2.x 用 weights= 枚举加载预训练权重，pretrained=True 已废弃
    model = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    model.fc = nn.Linear(2048, config.num_classes)
    return model
