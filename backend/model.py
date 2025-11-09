"""
Neural Network Model for Flower Classification v2.0
Supports both static images and live camera feed
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from config import FLOWER_CLASSES, IMAGE_SIZE, IMAGE_MEAN, IMAGE_STD, DEVICE

class FlowerCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(FlowerCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class FlowerClassifier:
    def __init__(self, model_path=None):
        self.device = DEVICE
        self.model = FlowerCNN(num_classes=len(FLOWER_CLASSES)).to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)
        ])
        if model_path:
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"Model loaded from {model_path}")
            except Exception as e:
                print(f"Warning: Could not load model: {e}")

    def preprocess_image(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            return image_tensor
        except Exception as e:
            raise ValueError(f"Error processing image: {e}")

    def predict(self, image_path):
        with torch.no_grad():
            image_tensor = self.preprocess_image(image_path)
            output = self.model(image_tensor)
            probabilities = F.softmax(output, dim=1)
            probs, indices = torch.topk(probabilities, 3, dim=1)
            predictions = {
                'flower': FLOWER_CLASSES[indices[0][0].item()],
                'confidence': float(probs[0][0].item()),
                'all_probabilities': {
                    FLOWER_CLASSES[i]: float(prob.item()) 
                    for i, prob in enumerate(probabilities[0])
                },
                'top_3': [
                    {
                        'flower': FLOWER_CLASSES[indices[0][i].item()],
                        'probability': float(probs[0][i].item())
                    }
                    for i in range(3)
                ]
            }
            return predictions
