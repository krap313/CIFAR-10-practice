import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm

# Custom CosineAnnealingWarmupRestarts Scheduler
class CosineAnnealingWarmupRestarts(_LRScheduler):
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 first_cycle_steps: int,
                 cycle_mult: float = 1.,
                 max_lr: float = 0.1,
                 min_lr: float = 0.001,
                 warmup_steps: int = 0,
                 gamma: float = 1.,
                 last_epoch: int = -1):
        assert warmup_steps < first_cycle_steps

        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.base_max_lr = max_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.gamma = gamma

        self.cur_cycle_steps = first_cycle_steps
        self.cycle = 0
        self.step_in_cycle = last_epoch

        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) * \
                    (1 + math.cos(math.pi * (self.step_in_cycle - self.warmup_steps) / \
                                    (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** n
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lr = self.base_max_lr * (self.gamma ** self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

# ResNet6 모델 정의
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
        self.fc = nn.Linear(64, num_classes)

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

        x = lam * x1 + (1 - lam) * x2
        y = lam * torch.nn.functional.one_hot(y1, num_classes=10).float() + \
            (1 - lam) * torch.nn.functional.one_hot(y2, num_classes=10).float()
        return x, y

# Label Smoothing Loss
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.smooth = smoothing / classes

    def forward(self, pred, target):
        if target.dim() > 1:
            smoothed_labels = target * self.confidence + self.smooth
        else:
            target = target.to(dtype=torch.int64)
            one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
            smoothed_labels = one_hot * self.confidence + self.smooth

        log_probs = nn.LogSoftmax(dim=1)(pred)
        loss = -(smoothed_labels * log_probs).sum(dim=1).mean()
        return loss

# 데이터 증강
stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
train_tfms = transforms.Compose([
    transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
    transforms.RandomHorizontalFlip(),
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
    train_size = int(len(trainset) * 0.9)
    train_subset, val_subset = random_split(
        trainset, [train_size, len(trainset) - train_size]
    )

    # DataLoader
    trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    valloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=8)

    # ResNet6 모델 불러오기
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    net = ResNet6(num_classes=10).to(device)

    # Optimizer, Scheduler, Loss
    optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=10, cycle_mult=2, max_lr=lr, min_lr=1e-5, warmup_steps=5, gamma=0.5)
    criterion = LabelSmoothingLoss(classes=10, smoothing=0.1)

    mixup = Mixup(alpha)

    # 학습
    train_losses, val_accuracies = [], []
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0

        with tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}") as t:
            for inputs, labels in t:
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
                t.set_postfix(loss=running_loss / len(t))

        train_losses.append(running_loss / len(trainloader))
        scheduler.step(epoch + 1)

        # 검증
        net.eval()
        correct, total = 0, 0
        with torch.no_grad():
            with tqdm(valloader, desc="Validation") as t:
                for inputs, labels in t:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = net(inputs)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    t.set_postfix(accuracy=correct / total)
        val_accuracy = correct / total
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_losses[-1]:.4f}, Val Accuracy: {val_accuracy:.4f}")

    return train_losses, val_accuracies

# 결과 시각화
def plot_results(epochs, train_losses, val_accuracies):
    plt.figure(figsize=(12, 6))

    # 손실 그래프
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, label="Train Loss")
    plt.title("Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")

    # 정확도 그래프
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), val_accuracies, label="Validation Accuracy")
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
    batch_size = 2048
    lr = 0.01
    epochs = 100
    alpha = 0.4  # Mixup의 알파 값

    # 학습 실행
    train_losses, val_accuracies = train_cifar(batch_size, lr, epochs, alpha, data_dir=data_dir)

    # 결과 시각화
    plot_results(epochs, train_losses, val_accuracies)

if __name__ == "__main__":
    main()
