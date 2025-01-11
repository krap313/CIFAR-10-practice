import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import random_split, DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.transforms import AutoAugment, AutoAugmentPolicy

# Residual Block (Same as before)
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

# ResNet6 with Dropout
class ResNet6(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet6, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(16, 16, 1, stride=1)
        self.layer2 = self._make_layer(16, 32, 2, stride=2)
        self.layer3 = self._make_layer(32, 64, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(0.3),  # Dropout for regularization
            nn.Linear(64, num_classes)
        )

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = [ResidualBlock(in_channels, out_channels, stride)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Data Augmentation with AutoAugment and RandomErasing
stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
train_tfms = transforms.Compose([
    AutoAugment(policy=AutoAugmentPolicy.CIFAR10),
    transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(*stats),
    transforms.RandomErasing(p=0.2)  # RandomErasing added
])
valid_tfms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(*stats)
])

# Load Data
def load_data(data_dir="./data"):
    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_tfms)
    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=valid_tfms)
    return trainset, testset

# Training Loop with OneCycleLR and Ranger Optimizer
def train_cifar(batch_size, lr, epochs, data_dir=None):
    trainset, testset = load_data(data_dir)
    train_size = int(len(trainset) * 0.9)
    train_subset, val_subset = random_split(trainset, [train_size, len(trainset) - train_size])

    trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet6(num_classes=10).to(device)

    # Ranger Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)  # Increased weight_decay

    # OneCycleLR Scheduler
    scheduler = OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(trainloader), epochs=epochs, pct_start=0.3)

    criterion = nn.CrossEntropyLoss()

    train_losses, val_accuracies = [], []
    lrs = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0

        for images, labels in tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()  # Update learning rate
            running_loss += loss.item()

        train_losses.append(running_loss / len(trainloader))
        lrs.append(scheduler.get_last_lr()[0])  # Log learning rate

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in valloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        val_accuracies.append(accuracy)

        print(f"Epoch {epoch+1}/{epochs} | Loss: {train_losses[-1]:.4f} | Accuracy: {accuracy:.4f}")

    # Plot Loss, Accuracy, and Learning Rate
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.title("Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.subplot(1, 3, 2)
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label="Validation Accuracy")
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.subplot(1, 3, 3)
    plt.plot(range(1, len(lrs) + 1), lrs, label="Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")

    plt.tight_layout()
    plt.show()

# Main
def main():
    train_cifar(batch_size=128, lr=0.01, epochs=50, data_dir="./data")

if __name__ == '__main__':
    main()
