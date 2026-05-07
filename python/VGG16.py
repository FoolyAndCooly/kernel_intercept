#!/usr/bin/env python3
"""
VGG16 模型包装
提供与 GPT.py 类似的接口，用于 Orion 调度测试
"""

import torch
import torch.nn as nn

class VGG16Model(nn.Module):
    """VGG16 模型包装类"""

    def __init__(self, num_classes=1000, use_torchvision=True):
        super(VGG16Model, self).__init__()

        if use_torchvision:
            try:
                from torchvision.models import vgg16
                self.model = vgg16(weights=None)
                print("Using torchvision VGG16")
            except ImportError:
                print("torchvision not available, using built-in VGG16")
                self.model = self._build_vgg16(num_classes)
        else:
            self.model = self._build_vgg16(num_classes)

    def _build_vgg16(self, num_classes):
        """构建简化版 VGG16"""
        return nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Block 4
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Block 5
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Classifier
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        return self.model(x)


# 默认配置（与 profile_gpt_vgg16.sh 保持一致）
BATCH_SIZE = 8
IMAGE_SIZE = 224
NUM_CLASSES = 1000
