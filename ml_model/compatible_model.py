#!/usr/bin/env python3
"""
Совместимая модель для Django
"""

import torch
import torch.nn as nn
import torchvision.models as models

class CompatibleCarModel(nn.Module):
    """Совместимая модель для Django."""
    
    def __init__(self, pretrained=True):
        super().__init__()
        
        # Используем предобученную ResNet18
        self.backbone = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        
        # Заменяем последний слой
        self.backbone.fc = nn.Linear(512, 512)
        
        # Головы для предсказаний
        self.clean_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        self.intact_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        features = self.backbone(x)
        
        clean = self.clean_head(features)
        intact = self.intact_head(features)
        
        return {
            'clean': clean,
            'intact': intact
        }
