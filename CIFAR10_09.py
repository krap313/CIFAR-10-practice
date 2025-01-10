import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader
import numpy as np
from torchvision.models import resnet50, ResNet50_Weights
import matplotlib.pyplot as plt
from PIL import Image

## ResNet-50 모델 초기화
net = resnet50(weights=ResNet50_Weights.DEFAULT)
net.fc = nn.Linear(net.fc.in_features, 10)

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
    def __init__(self, std=0.01):
        self.std = std

    def __call__(self, img):
        if not isinstance(img, torch.Tensor):
            img = transforms.ToTensor()(img)
        noise = torch.randn_like(img) * self.std
        img = img + noise
        img = torch.clamp(img, 0.0, 1.0)
        return transforms.ToPILImage()(img)

# Mixup 데이터 증강
class Mixup:
    def __init__(self, alpha=0.4):
        self.alpha = alpha

    def __call__(self, x1, y1, x2, y2):
        lam = np.random.beta(self.alpha, self.alpha)
        y1 = y1.to(dtype=torch.int64)
        y2 = y2.to(dtype=torch.int64)

        assert torch.max(y1) < 10 and torch.max(y2) < 10, "Mixup target values are out of range"

        x = lam * x1 + (1 - lam) * x2
        y = lam * torch.nn.functional.one_hot(y1, num_classes=10).to(x1.device).float() + \
            (1 - lam) * torch.nn.functional.one_hot(y2, num_classes=10).to(x1.device).float()
        return x, y

# Label Smoothing Loss
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.smooth = smoothing / classes

    def forward(self, pred, target):
        target = target.to(dtype=torch.int64)

        if target.dim() > 1:
            target = target.argmax(dim=1)  # 다차원 타겟을 1차원으로 변환

        assert torch.max(target) < pred.size(1), f"Target indices are out of range. Max target value: {torch.max(target)}, pred.size(1): {pred.size(1)}"

        if target.dim() == 1:
            target = target.view(-1, 1)  # 1차원 타겟을 2차원으로 변환

        one_hot = torch.zeros_like(pred).scatter(1, target, 1).to(pred.device)

        smoothed_labels = one_hot * self.confidence + self.smooth
        log_probs = nn.LogSoftmax(dim=1)(pred)
        loss = -(smoothed_labels * log_probs).sum(dim=1).mean()
        return loss

# 데이터 증강
stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
train_tfms = transforms.Compose([
    transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    Cutout(size=8),
    AddNoise(std=0.01),
    transforms.ToTensor(),
    transforms.Normalize(*stats)
])
valid_tfms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(*stats)
])

# 데이터 로드 함수
def load_data(data_dir="./data"):
    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_tfms
    )
    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=valid_tfms
    )
    return trainset, testset

# 학습 함수
def train_cifar(batch_size, lr, epochs, alpha, data_dir=None):
    trainset, testset = load_data(data_dir)

    # 데이터 분할
    train_size = int(len(trainset) * 0.8)
    train_subset, val_subset = random_split(
        trainset, [train_size, len(trainset) - train_size]
    )

    # DataLoader
    trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
    valloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)

    # 사전 학습된 ResNet-50 불러오기
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    net = resnet50(weights=ResNet50_Weights.DEFAULT)
    net.fc = nn.Linear(net.fc.in_features, 10)
    net.to(device)

    # Optimizer, Scheduler, Loss
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = LabelSmoothingLoss(classes=10, smoothing=0.1)

    mixup = Mixup(alpha)

    # 학습
    train_losses, val_accuracies = [], []
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Mixup 데이터 증강
            if np.random.rand() > 0.5:
                indices = torch.randperm(inputs.size(0)).to(device)
                mix_inputs, mix_labels = mixup(inputs, labels, inputs[indices], labels[indices])
            else:
                mix_inputs, mix_labels = inputs, torch.nn.functional.one_hot(labels, num_classes=10).to(device).float()

            optimizer.zero_grad()
            outputs = net(mix_inputs)
            loss = criterion(outputs, mix_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()  # 각 배치 손실 합산

        train_losses.append(running_loss / len(trainloader))
        scheduler.step()

        # 검증
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

        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_losses[-1]:.4f}, Val Accuracy: {val_accuracy:.4f}")

    return train_losses, val_accuracies

# 결과 시각화
def plot_results(epochs, train_losses, val_accuracies):
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
    plt.savefig("training_results.png")
    plt.show()

# Main 함수
def main():
    data_dir = os.path.abspath("./data")
    batch_size = 256
    lr = 0.01
    epochs = 30
    alpha = 0.4  # Mixup의 알파 값

    # 학습 실행
    train_losses, val_accuracies = train_cifar(batch_size, lr, epochs, alpha, data_dir=data_dir)

    # 결과 시각화
    plot_results(range(1, epochs + 1), train_losses, val_accuracies)

if __name__ == "__main__":
    main()
