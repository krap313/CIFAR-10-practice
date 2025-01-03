import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm  # For progress bar
import json  # For saving results

# Data transforms (normalization & data augmentation)
stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

def get_transforms(normalization=True, data_augmentation=True):
    train_tfms = []
    if data_augmentation:
        train_tfms.extend([
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10)
        ])
    train_tfms.append(transforms.ToTensor())
    if normalization:
        train_tfms.append(transforms.Normalize(*stats))

    valid_tfms = [transforms.ToTensor()]
    if normalization:
        valid_tfms.append(transforms.Normalize(*stats))

    return transforms.Compose(train_tfms), transforms.Compose(valid_tfms)

# Load CIFAR-10 data
def load_data(data_dir="./data", normalization=True, data_augmentation=True):
    train_tfms, valid_tfms = get_transforms(normalization, data_augmentation)

    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_tfms
    )

    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=valid_tfms
    )

    return trainset, testset

# Define the neural network with Residual Connections
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

class SimpleCNN(nn.Module):
    def __init__(self, batch_norm=False):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16) if batch_norm else nn.Identity()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32) if batch_norm else nn.Identity()
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64) if batch_norm else nn.Identity()
        self.fc = nn.Linear(64 * 32 * 32, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Training function
def train_cifar(model, batch_size, lr, epochs, data_dir, normalization, data_augmentation, lr_scheduling, optimizer_type, gradient_clipping):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    if optimizer_type == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    trainset, testset = load_data(data_dir, normalization, data_augmentation)

    train_size = int(len(trainset) * 0.8)
    train_subset, val_subset = random_split(
        trainset, [train_size, len(trainset) - train_size]
    )

    trainloader = torch.utils.data.DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    valloader = torch.utils.data.DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    scheduler = None
    if lr_scheduling:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=len(trainloader))

    epoch_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            if gradient_clipping:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler:
                scheduler.step()
            running_loss += loss.item()

        epoch_losses.append(running_loss / len(trainloader))

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in valloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        val_accuracies.append(accuracy)

    return max(val_accuracies)

# Plot results
def plot_results(results):
    labels = [str(res['experiment']) for res in results]
    accuracies = [res['accuracy'] for res in results]

    plt.figure(figsize=(10, 6))
    plt.bar(labels, accuracies, color='skyblue')
    plt.xlabel('Experiment Configurations')
    plt.ylabel('Validation Accuracy')
    plt.title('Accuracy Comparison Across Experiments')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# Experiment Runner
def run_experiments():
    experiments = [
        {"normalization": False, "resnet": False, "lr_scheduling": False, "data_augmentation": False, "batch_norm": False, "optimizer": "SGD", "gradient_clipping": False},
        {"normalization": True, "resnet": False, "lr_scheduling": False, "data_augmentation": False, "batch_norm": False, "optimizer": "SGD", "gradient_clipping": False},
        {"normalization": True, "resnet": True, "lr_scheduling": False, "data_augmentation": False, "batch_norm": True, "optimizer": "SGD", "gradient_clipping": False},
        {"normalization": True, "resnet": True, "lr_scheduling": True, "data_augmentation": True, "batch_norm": True, "optimizer": "Adam", "gradient_clipping": True},
    ]

    results = []

    for i, exp in enumerate(tqdm(experiments, desc="Running Experiments")):
        print(f"Running experiment {i+1}/{len(experiments)}: {exp}")
        model = ResNet() if exp["resnet"] else SimpleCNN(batch_norm=exp["batch_norm"])
        accuracy = train_cifar(
            model=model,
            batch_size=128,
            lr=1e-3,
            epochs=10,
            data_dir="./data",
            normalization=exp["normalization"],
            data_augmentation=exp["data_augmentation"],
            lr_scheduling=exp["lr_scheduling"],
            optimizer_type=exp["optimizer"],
            gradient_clipping=exp["gradient_clipping"]
        )
        results.append({"experiment": exp, "accuracy": accuracy})
        print(f"Experiment {i+1}/{len(experiments)} completed. Accuracy: {accuracy:.4f}")

    # Save results to a file
    with open("experiment_results.json", "w") as f:
        json.dump(results, f, indent=4)

    for res in results:
        print(f"Experiment: {res['experiment']}, Accuracy: {res['accuracy']:.4f}")

    plot_results(results)

if __name__ == "__main__":
    run_experiments()
