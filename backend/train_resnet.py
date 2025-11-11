"""
train_resnet.py
Transfer learning with ResNet18 for 5-class flower classification.
Place this file in backend/ and run: python3 train_resnet.py
"""

import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from config import IMAGE_SIZE, IMAGE_MEAN, IMAGE_STD, DEVICE, FLOWER_CLASSES
import sys

# ---------- Config ----------
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'flowers')
BATCH_SIZE = 32
NUM_EPOCHS = 12            # start here; increase if needed
LR = 1e-3
WEIGHT_DECAY = 1e-4
NUM_CLASSES = len(FLOWER_CLASSES)
MODEL_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'flower_model_resnet.pth')
os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)

# Choose safe num_workers for DataLoader on macOS / M1: 0 (no multiprocessing) or small number
DEFAULT_NUM_WORKERS = 0 if sys.platform == 'darwin' else 4

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total = 0
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data).item()
        total += labels.size(0)
    epoch_loss = running_loss / total
    epoch_acc = 100.0 * running_corrects / total
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data).item()
            total += labels.size(0)
    epoch_loss = running_loss / total
    epoch_acc = 100.0 * running_corrects / total
    return epoch_loss, epoch_acc

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training device: {device}")
    print("Loading dataset from:", DATA_DIR)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
    ])

    dataset = datasets.ImageFolder(DATA_DIR, transform=train_transform)
    print("Total images:", len(dataset))

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    # ensure validation uses deterministic transform
    val_dataset.dataset.transform = val_transform

    num_workers = DEFAULT_NUM_WORKERS
    print(f"Using num_workers={num_workers} for DataLoader (platform: {sys.platform})")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=num_workers, pin_memory=(device.type == 'cuda'))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=max(0, num_workers//2), pin_memory=(device.type == 'cuda'))

    print("Loading pre-trained ResNet18...")
    model = models.resnet18(pretrained=True)

    # Freeze all layers initially
    for param in model.parameters():
        param.requires_grad = False

    # Replace fc
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)

    model = model.to(device)

    # Only parameters of final layer are being optimized
    optimizer = optim.Adam(model.fc.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    since = time.time()

    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        t1 = time.time()
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
              f"TL: {train_loss:.4f} | TA: {train_acc:.2f}% | "
              f"VL: {val_loss:.4f} | VA: {val_acc:.2f}% | "
              f"time: {int(t1-t0)}s")

        # deep copy best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    # Save best model
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), MODEL_OUT)
    elapsed = time.time() - since
    print(f"Training complete in {int(elapsed // 60)}m {int(elapsed % 60)}s")
    print(f"Best val acc: {best_acc:.2f}%")
    print(f"Saved best model to: {MODEL_OUT}")

if __name__ == "__main__":
    main()
