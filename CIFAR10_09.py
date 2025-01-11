import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader
import numpy as np
from torchvision.models import resnet101, ResNet101_Weights
import matplotlib.pyplot as plt
from PIL import Image


## 개선점
# 이미 학습된 ResNet-101 가져옴 - 이놈이 문제(너무 복잡한 함수, layer를 낮추자)
# 데이터 증강 방식 추가(mixup)
# label smoothing -> 일반화 분포 학습, 과적합 방지
# Cosine Annealing Warm Restarts
# epoch 30 -> 50

# linear warm up 적용(초반 학습률)
# cosine annealing warmup때문에 문제 발생(학습률 최댓값, 주기 너무 짧을수도)


## Result
# Epoch 1/50, Loss: 1.6252, Val Accuracy: 0.1416
# Epoch 2/50, Loss: 1.3841, Val Accuracy: 0.3002
# Epoch 3/50, Loss: 1.3250, Val Accuracy: 0.6694
# Epoch 4/50, Loss: 1.2845, Val Accuracy: 0.4807
# Epoch 5/50, Loss: 1.2638, Val Accuracy: 0.6669
# Epoch 6/50, Loss: 1.2368, Val Accuracy: 0.5067
# Epoch 7/50, Loss: 1.1879, Val Accuracy: 0.3696
# Epoch 8/50, Loss: 1.1784, Val Accuracy: 0.5675
# Epoch 9/50, Loss: 1.1689, Val Accuracy: 0.6960
# Epoch 10/50, Loss: 1.1424, Val Accuracy: 0.6647
# Epoch 11/50, Loss: 1.3085, Val Accuracy: 0.1408
# Epoch 12/50, Loss: 1.2716, Val Accuracy: 0.1045
# Epoch 13/50, Loss: 1.2687, Val Accuracy: 0.5891
# Epoch 14/50, Loss: 1.2566, Val Accuracy: 0.6989
# Epoch 15/50, Loss: 1.2421, Val Accuracy: 0.6491
# Epoch 16/50, Loss: 1.2940, Val Accuracy: 0.1059
# Epoch 17/50, Loss: 1.3043, Val Accuracy: 0.4413
# Epoch 18/50, Loss: 1.2657, Val Accuracy: 0.6032
# Epoch 19/50, Loss: 1.2362, Val Accuracy: 0.5604
# Epoch 20/50, Loss: 1.2085, Val Accuracy: 0.6321
# Epoch 21/50, Loss: 1.2001, Val Accuracy: 0.6269
# Epoch 22/50, Loss: 1.1887, Val Accuracy: 0.2259
# Epoch 23/50, Loss: 1.1610, Val Accuracy: 0.7468
# Epoch 24/50, Loss: 1.1637, Val Accuracy: 0.6876
# Epoch 25/50, Loss: 1.1524, Val Accuracy: 0.7137
# Epoch 26/50, Loss: 1.1247, Val Accuracy: 0.7526
# Epoch 27/50, Loss: 1.1128, Val Accuracy: 0.7536
# Epoch 28/50, Loss: 1.0839, Val Accuracy: 0.7283
# Epoch 29/50, Loss: 1.0930, Val Accuracy: 0.7189
# Epoch 30/50, Loss: 1.0826, Val Accuracy: 0.7480
# Epoch 31/50, Loss: 1.1834, Val Accuracy: 0.1131
# Epoch 32/50, Loss: 1.2171, Val Accuracy: 0.5149
# Epoch 33/50, Loss: 1.2148, Val Accuracy: 0.4792
# Epoch 34/50, Loss: 1.2420, Val Accuracy: 0.6843
# Epoch 35/50, Loss: 1.2015, Val Accuracy: 0.4813
# Epoch 36/50, Loss: 1.2075, Val Accuracy: 0.1621
# Epoch 37/50, Loss: 1.2115, Val Accuracy: 0.6150
# Epoch 38/50, Loss: 1.1751, Val Accuracy: 0.7035
# Epoch 39/50, Loss: 1.1109, Val Accuracy: 0.1379
# Epoch 40/50, Loss: 1.1611, Val Accuracy: 0.3659
# Epoch 41/50, Loss: 1.1806, Val Accuracy: 0.6793
# Epoch 42/50, Loss: 1.1675, Val Accuracy: 0.5780
# Epoch 43/50, Loss: 1.1225, Val Accuracy: 0.6776
# Epoch 44/50, Loss: 1.1225, Val Accuracy: 0.7208
# Epoch 45/50, Loss: 1.1186, Val Accuracy: 0.2068
# Epoch 46/50, Loss: 1.0694, Val Accuracy: 0.3731
# Epoch 47/50, Loss: 1.1083, Val Accuracy: 0.4530
# Epoch 48/50, Loss: 1.1036, Val Accuracy: 0.6581
# Epoch 49/50, Loss: 1.0069, Val Accuracy: 0.1257
# Epoch 50/50, Loss: 1.0648, Val Accuracy: 0.6792



## ResNet-101 모델 초기화
net = resnet101(weights=ResNet101_Weights.DEFAULT)
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

    # 사전 학습된 ResNet-101 불러오기
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    net = resnet101(weights=ResNet101_Weights.DEFAULT)
    net.fc = nn.Linear(net.fc.in_features, 10)
    net.to(device)

    # Optimizer, Scheduler, Loss
    optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
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
        scheduler.step(epoch + 1)

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
    lr = 0.001
    epochs = 50
    alpha = 0.4  # Mixup의 알파 값

    # 학습 실행
    train_losses, val_accuracies = train_cifar(batch_size, lr, epochs, alpha, data_dir=data_dir)

    # 결과 시각화
    plot_results(range(1, epochs + 1), train_losses, val_accuracies)

if __name__ == "__main__":
    main()


