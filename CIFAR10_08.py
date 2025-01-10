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

# 추가된 증강 방식에 맞게 증강 강도 및 lr, optimizer 조정

## Result
# Epoch 1/30, Loss: 1.6248, Val Accuracy: 0.4379
# Epoch 2/30, Loss: 1.2524, Val Accuracy: 0.5709
# Epoch 3/30, Loss: 1.0655, Val Accuracy: 0.5787
# Epoch 4/30, Loss: 0.9205, Val Accuracy: 0.5489
# Epoch 5/30, Loss: 0.8275, Val Accuracy: 0.6168
# Epoch 6/30, Loss: 0.7477, Val Accuracy: 0.6724
# Epoch 7/30, Loss: 0.6877, Val Accuracy: 0.7300
# Epoch 8/30, Loss: 0.6453, Val Accuracy: 0.7556
# Epoch 9/30, Loss: 0.6051, Val Accuracy: 0.7361
# Epoch 10/30, Loss: 0.5708, Val Accuracy: 0.7682
# Epoch 11/30, Loss: 0.5524, Val Accuracy: 0.7719
# Epoch 12/30, Loss: 0.5244, Val Accuracy: 0.7542
# Epoch 13/30, Loss: 0.5027, Val Accuracy: 0.7842
# Epoch 14/30, Loss: 0.4907, Val Accuracy: 0.7902
# Epoch 15/30, Loss: 0.4764, Val Accuracy: 0.7925
# Epoch 16/30, Loss: 0.4578, Val Accuracy: 0.8022
# Epoch 17/30, Loss: 0.4496, Val Accuracy: 0.7693
# Epoch 18/30, Loss: 0.4326, Val Accuracy: 0.7807
# Epoch 19/30, Loss: 0.4178, Val Accuracy: 0.8108
# Epoch 20/30, Loss: 0.3975, Val Accuracy: 0.8410
# Epoch 21/30, Loss: 0.3697, Val Accuracy: 0.8385
# Epoch 22/30, Loss: 0.3559, Val Accuracy: 0.8490
# Epoch 23/30, Loss: 0.3321, Val Accuracy: 0.8550
# Epoch 24/30, Loss: 0.2983, Val Accuracy: 0.8640
# Epoch 25/30, Loss: 0.2663, Val Accuracy: 0.8779
# Epoch 26/30, Loss: 0.2342, Val Accuracy: 0.8850
# Epoch 27/30, Loss: 0.1835, Val Accuracy: 0.8935
# Epoch 28/30, Loss: 0.1582, Val Accuracy: 0.9021
# Epoch 29/30, Loss: 0.1289, Val Accuracy: 0.9068
# Epoch 30/30, Loss: 0.1223, Val Accuracy: 0.9093

# Cutout 클래스 정의
class Cutout:
    def __init__(self, size):
        self.size = size  # 잘라낼 사각형의 크기

    def __call__(self, img):
        h, w = img.size
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        x1 = np.clip(x - self.size // 2, 0, w)
        y1 = np.clip(y - self.size // 2, 0, h)
        x2 = np.clip(x + self.size // 2, 0, w)
        y2 = np.clip(y + self.size // 2, 0, h)
        img = np.array(img)
        img[y1:y2, x1:x2] = 0  # 해당 영역을 검은색으로 만듦
        return Image.fromarray(img)


# AddNoise 클래스 정의
class AddNoise:
    def __init__(self, std=0.02):  # 노이즈 강도 감소
        self.std = std

    def __call__(self, img):
        if not isinstance(img, torch.Tensor):
            img = transforms.ToTensor()(img)
        noise = torch.randn_like(img) * self.std
        img = img + noise
        img = torch.clamp(img, 0.0, 1.0)  # 값 범위 유지
        return transforms.ToPILImage()(img)


# 데이터 증강 및 정규화
stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
train_tfms = transforms.Compose([
    transforms.RandomCrop(32, padding=4, padding_mode='reflect'),  # 랜덤 크롭
    transforms.RandomHorizontalFlip(),  # 좌우 반전
    transforms.RandomRotation(degrees=10),  # 회전 강도 감소
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),  # 색상 변화 강도 감소
    Cutout(size=4),  # Cutout 크기 축소
    AddNoise(std=0.02),  # 노이즈 강도 감소
    transforms.ToTensor(),  # 텐서로 변환
    transforms.Normalize(*stats)  # 정규화
])
valid_tfms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(*stats)
])


# CIFAR-10 데이터 로드
def load_data(data_dir="./data"):
    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_tfms
    )
    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=valid_tfms
    )
    return trainset, testset


# ResNet 모델 정의 (간소화)
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


# 학습 함수 정의
def train_cifar(batch_size, lr, epochs, data_dir=None):
    trainset, testset = load_data(data_dir)

    # 데이터 분할
    train_size = int(len(trainset) * 0.8)
    train_subset, val_subset = random_split(
        trainset, [train_size, len(trainset) - train_size]
    )

    # DataLoader
    trainloader = torch.utils.data.DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    valloader = torch.utils.data.DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    # 모델 및 학습 설정
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    net = ResNet(num_classes=10).to(device)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=len(trainloader))
    criterion = nn.CrossEntropyLoss()

    # 학습 루프
    train_losses, val_accuracies = [], []
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

        # 학습 손실 저장
        train_losses.append(running_loss / len(trainloader))

        # 검증 정확도
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

    return train_losses, val_accuracies


# 학습 결과 시각화
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


# 증강된 이미지 시각화 함수
def visualize_augmented_images(dataset, n=6):
    plt.figure(figsize=(12, 4))
    for i in range(n):
        img, label = dataset[i]
        plt.subplot(1, n, i + 1)
        plt.imshow(img.permute(1, 2, 0).numpy())
        plt.axis('off')
        plt.title(f"Label: {label}")
    plt.tight_layout()
    plt.show()


# Main 함수 실행
def main():
    data_dir = os.path.abspath("./data")
    trainset, _ = load_data(data_dir)
    visualize_augmented_images(trainset)  # 증강된 이미지 확인

    batch_size = 128
    lr = 0.1
    epochs = 30

    # 학습 실행
    train_losses, val_accuracies = train_cifar(batch_size, lr, epochs, data_dir=data_dir)

    # 결과 시각화
    plot_results(range(1, epochs + 1), train_losses, val_accuracies)


if __name__ == "__main__":
    main()
