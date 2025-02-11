import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Cutout 클래스 정의
class Cutout:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        h, w = img.size
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        x1 = np.clip(x - self.size // 2, 0, w)
        y1 = np.clip(y - self.size // 2, 0, h)
        x2 = np.clip(x + self.size // 2, 0, w)
        y2 = np.clip(y + self.size // 2, 0, h)
        img = np.array(img)
        img[y1:y2, x1:x2] = 0
        return Image.fromarray(img)

# AddNoise 클래스 정의
class AddNoise:
    def __init__(self, std=0.02):
        self.std = std

    def __call__(self, img):
        if not isinstance(img, torch.Tensor):
            img = transforms.ToTensor()(img)
        noise = torch.randn_like(img) * self.std
        img = img + noise
        img = torch.clamp(img, 0.0, 1.0)
        return transforms.ToPILImage()(img)

# ResidualBlock 클래스 정의
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        return self.relu(out)

# 데이터 증강 및 정규화
stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
train_tfms = transforms.Compose([
    transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=30),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    Cutout(size=8),  # Increased size for better effect
    AddNoise(std=0.01),  # Reduced noise for cleaner augmentation
    transforms.ToTensor(),
    transforms.Normalize(*stats)
])
valid_tfms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(*stats)
])

# CIFAR-10 데이터 로드
def load_data(data_dir="./data"):
    try:
        trainset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=train_tfms
        )
        testset = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=valid_tfms
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise
    return trainset, testset

# ResNet 모델 정의
class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = [ResidualBlock(in_channels, out_channels, stride)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Linear Warmup Scheduler
class LinearWarmupScheduler:
    def __init__(self, optimizer, warmup_steps):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.current_step = 0

    def step(self):
        self.current_step += 1
        lr_scale = min(self.current_step / self.warmup_steps, 1.0)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr_scale * param_group['initial_lr']

# 학습 함수 정의
def train_cifar(batch_size, lr, epochs, data_dir=None):
    trainset, testset = load_data(data_dir)

    train_size = int(len(trainset) * 0.8)
    train_subset, val_subset = random_split(
        trainset, [train_size, len(trainset) - train_size]
    )

    try:
        trainloader = torch.utils.data.DataLoader(
            train_subset, batch_size=batch_size, shuffle=True, num_workers=4
        )
        valloader = torch.utils.data.DataLoader(
            val_subset, batch_size=batch_size, shuffle=False, num_workers=4
        )
    except Exception as e:
        print(f"Error initializing DataLoader: {e}")
        raise

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    net = ResNet(num_classes=10).to(device)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = lr

    warmup_scheduler = LinearWarmupScheduler(optimizer, warmup_steps=int(len(trainloader) * 5))  # Warmup for 5 epochs
    onecycle_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=len(trainloader))
    criterion = nn.CrossEntropyLoss()

    train_losses, val_accuracies = [], []

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0

        for step, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if epoch < 5:  # Apply linear warmup for the first 5 epochs
                warmup_scheduler.step()
            else:  # Use OneCycleLR scheduler afterwards
                onecycle_scheduler.step()

            running_loss += loss.item()

        train_losses.append(running_loss / len(trainloader))

        net.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in valloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_accuracy = correct / total
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(trainloader):.4f}, Val Accuracy: {val_accuracy:.4f}")

        # 실시간 그래프 업데이트
        plot_results(range(1, epoch + 2), train_losses, val_accuracies, "training_results.png")

    return train_losses, val_accuracies

# 학습 결과 시각화
def plot_results(epochs, train_losses, val_accuracies, filename):
    epochs = range(1, len(train_losses) + 1)  # Adjust epochs to match train_losses and val_accuracies
    plt.figure(figsize=(12, 6))

    # 손실 그래프
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.title("Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")

    # 정확도 그래프
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, label="Validation Accuracy")
    plt.title("Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Main 함수 실행
def main():
    data_dir = os.path.abspath("./data")
    batch_size = 128
    lr = 0.05  # Reduced learning rate for smoother training
    epochs = 50

    train_losses, val_accuracies = train_cifar(batch_size, lr, epochs, data_dir=data_dir)

if __name__ == "__main__":
    main()