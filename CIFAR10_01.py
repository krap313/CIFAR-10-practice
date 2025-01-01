import ssl
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

# Test 01
# bash size: 32
# learning rate(lr): 0.001
# epoch: 10

# Test loss: 0.0324
# Test accuracy: 64.33%

# Test 02
# bash size: 32
# learning rate(lr): 0.0005
# epoch: 20

# Test loss: 0.0324
# Test accuracy: 64.33%

if __name__ == "__main__":

    # train dataset download
    train_dataset = datasets.CIFAR10(root="./data/", train=True, download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    print()
    print("---Train dataset---")
    print(train_loader.dataset)

    # test dataset download
    test_dataset = datasets.CIFAR10(root="./data/", train=False, download=True, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)
    print()
    print("---Test dataset---")
    print(test_loader.dataset)

    # 입력 data와 label 데이터 size 및 type 확인
    for (X_train, Y_train) in train_loader:
        print(f"X_train: {X_train.size()}  |   type: {X_train.type()}")
        print(f"Y_train: {Y_train.size()}  |   type: {Y_train.type()}")
        break

    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.conv1 = nn.Conv2d(
                in_channels=3,
                out_channels=8,
                kernel_size=3,
                padding=1
            )
            self.conv2 = nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=3,
                padding=1
            )
            self.pool = nn.MaxPool2d(
                kernel_size=2,
                stride=2
            )
            self.fc1 = nn.Linear(8 * 8 * 16, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = torch.relu(x)
            x = self.pool(x)
            x = self.conv2(x)
            x = torch.relu(x)
            x = self.pool(x)

            x = x.view(-1, 8 * 8 * 16)
            x = self.fc1(x)
            x = torch.relu(x)
            x = self.fc2(x)
            x = torch.relu(x)
            x = self.fc3(x)
            x = torch.log_softmax(x, dim=1)
            return x

    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print(f"Using PyTorch version: {torch.__version__}  |  Device: {DEVICE}")

    model = CNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    test_losses = []
    test_accuracies = []

    def train(model, train_loader, optimizer, log_interval):
        model.train()
        for batch_idx, (image, label) in enumerate(train_loader):
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            if batch_idx % log_interval == 0:
                print(f"train Epoch: {Epoch} [{batch_idx * len(image)}/{len(train_loader.dataset)}({100. * batch_idx / len(train_loader):.0f}%)]\tTrain Loss: {loss.item()}")
                train_losses.append(loss.item())

    def evaluate(model, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for image, label in test_loader:
                image = image.to(DEVICE)
                label = label.to(DEVICE)
                output = model(image)
                test_loss += criterion(output, label).item()
                prediction = output.max(1, keepdim=True)[1]
                correct += prediction.eq(label.view_as(prediction)).sum().item()

        test_loss /= len(test_loader.dataset)
        test_accuracy = 100. * correct / len(test_loader.dataset)
        return test_loss, test_accuracy

    EPOCHS = 20
    for Epoch in range(1, EPOCHS + 1):
        train(model, train_loader, optimizer, log_interval=200)
        test_loss, test_accuracy = evaluate(model, test_loader)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        print(f"\n[EPOCH: {Epoch}]\tTest Loss: {test_loss:.4f}\tTest Accuracy: {test_accuracy} % \n")

    # Plotting results
    epochs_range = range(1, EPOCHS + 1)

    plt.figure(figsize=(12, 6))

    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, test_losses, label="Test Loss")
    plt.title("Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")

    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, test_accuracies, label="Test Accuracy")
    plt.title("Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend(loc="lower right")

    plt.tight_layout()

    # Save plots
    plt.savefig("training_results.png")
    print("Plots saved as 'training_results.png'")

    plt.show()
