import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Data transforms (normalization & data augmentation)
stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
train_tfms = transforms.Compose([
    transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(*stats)
])
valid_tfms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(*stats)
])

# Load CIFAR-10 data
def load_data(data_dir="./data"):
    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_tfms
    )

    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=valid_tfms
    )

    return trainset, testset

# Define the neural network
class Net(nn.Module):
    def __init__(self, l1=120, l2=84):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, l1)
        self.fc2 = nn.Linear(l1, l2)
        self.fc3 = nn.Linear(l2, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Training function
def train_cifar(l1, l2, lr, batch_size, epochs, data_dir=None):
    net = Net(l1, l2)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)

    trainset, testset = load_data(data_dir)

    train_size = int(len(trainset) * 0.8)
    train_subset, val_subset = random_split(
        trainset, [train_size, len(trainset) - train_size]
    )

    trainloader = torch.utils.data.DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    valloader = torch.utils.data.DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    epoch_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_value_(net.parameters(), 0.1)  # Gradient Clipping
            optimizer.step()
            running_loss += loss.item()

        epoch_losses.append(running_loss / len(trainloader))

        net.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in valloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        val_accuracies.append(accuracy)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {val_loss/len(valloader):.4f}, Accuracy: {accuracy:.4f}")

    return epoch_losses, val_accuracies

# Test accuracy
def test_accuracy(net, data_dir, device="cpu"):
    _, testset = load_data(data_dir)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2
    )

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total

# Plotting results
def plot_results(epochs, train_losses, val_accuracies):
    plt.figure(figsize=(12, 6))

    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.title("Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")

    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, label="Validation Accuracy")
    plt.title("Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig("training_results.png")
    print("Plots saved as 'training_results.png'")
    plt.show()

# Main function
def main():
    data_dir = os.path.abspath("./data")
    l1 = 120
    l2 = 84
    lr = 1e-3
    batch_size = 64
    epochs = 10

    train_losses, val_accuracies = train_cifar(l1, l2, lr, batch_size, epochs, data_dir=data_dir)

    # Plot results
    plot_results(range(1, epochs + 1), train_losses, val_accuracies)

if __name__ == "__main__":
    main()
