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

# ResNet-101 모델 초기화
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
            target = target.argmax(dim=1)

        assert torch.max(target) < pred.size(1), f"Target indices are out of range. Max target value: {torch.max(target)}, pred.size(1): {pred.size(1)}"

        if target.dim() == 1:
            target = target.view(-1, 1)

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
    train_size = int(len(trainset) * 0.9)
    train_subset, val_subset = random_split(
        trainset, [train_size, len(trainset) - train_size]
    )

    # DataLoader
    trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4)
    valloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 사전 학습된 ResNet-101 불러오기
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    net = resnet101(weights=ResNet101_Weights.DEFAULT)
    net.fc = nn.Linear(net.fc.in_features, 10)
    net.to(device)

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

        if epoch < 80:
            scheduler.step(epoch + 1)

        if epoch == 79:
            print("[INFO] Learning rate schedule is now fixed.")

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
    batch_size = 128
    lr = 0.01
    epochs = 120
    alpha = 0.4  # Mixup의 알파 값

    # 학습 실행
    train_losses, val_accuracies = train_cifar(batch_size, lr, epochs, alpha, data_dir=data_dir)

    # 결과 시각화
    plot_results(epochs, train_losses, val_accuracies)

if __name__ == "__main__":
    main()