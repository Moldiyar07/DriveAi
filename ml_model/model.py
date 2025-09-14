import torch
import torch.nn as nn
from torchvision import models

class CarConditionModel(nn.Module):
    """Модель для определения чистоты и целостности машин."""
    
    def __init__(self, num_classes=2):
        super(CarConditionModel, self).__init__()
        
        # Используем ResNet18 как backbone
        self.backbone = models.resnet18(pretrained=True)
        
        # Заменяем последний слой
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # Убираем последний слой
        
        # Добавляем наши головы
        self.clean_head = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        self.intact_head = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        features = self.backbone(x)
        
        clean_prob = self.clean_head(features)
        intact_prob = self.intact_head(features)
        
        return clean_prob, intact_prob