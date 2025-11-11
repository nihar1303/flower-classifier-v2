"""
Training script for Flower Classification v2.0
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
import os
from model import FlowerCNN
from config import IMAGE_SIZE, IMAGE_MEAN, IMAGE_STD, DEVICE, FLOWER_CLASSES

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = total_loss / len(train_loader)
    return avg_loss, accuracy

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = total_loss / len(val_loader)
    return avg_loss, accuracy

def main():
    print("=" * 50)
    print("Flower Classification v2.0 - Training Script")
    print("=" * 50)

    BATCH_SIZE = 32
    EPOCHS = 20
    LEARNING_RATE = 0.001

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)
    ])

    print("Loading dataset...")
    try:
        dataset = datasets.ImageFolder('../data/flowers', transform=transform)
        print(f"✓ Dataset loaded: {len(dataset)} images")
    except FileNotFoundError:
        print("✗ Dataset not found at data/flowers/")
        return

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = DEVICE
    model = FlowerCNN(num_classes=len(FLOWER_CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Training on {device}...")
    print("=" * 50)

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{EPOCHS} | TL: {train_loss:.4f} | TA: {train_acc:.2f}% | VL: {val_loss:.4f} | VA: {val_acc:.2f}%")

    os.makedirs('models', exist_ok=True)
    model_path = 'models/flower_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"✓ Model saved to {model_path}")
    print("=" * 50)

if __name__ == '__main__':
    main()
