# %%
import torch
import torchvision
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
import matplotlib.pyplot as plt


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def plot_train_images(trainloader):
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    imshow(torchvision.utils.make_grid(images))


# Hyperparameters
learning_rate = 0.01
momentum = 0.9
num_epochs = 4
batch_size = 16

# Architecture
num_features = 3072  # 3x32x32
num_hidden_1 = 512
num_hidden_2 = 256
num_classes = 10


# Load data
transf = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transf
)
testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transf
)

train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=2
)
test_loader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=2
)

images, labels = iter(train_loader).next()
print(f"Image batch dim: {images.shape} \nLabel dim: {labels.shape}")



# Model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = torch.nn.Linear(num_features, num_hidden_1)
        self.fc2 = torch.nn.Linear(num_hidden_1, num_hidden_2)
        self.out = torch.nn.Linear(num_hidden_2, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.out(x)
        probas = F.log_softmax(logits, dim=1)
        return logits, probas

torch.manual_seed(0)
np.random.seed(0)
model = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

# %%
def compute_acc(net, data_loader):
    net.eval()
    correct_pred, num_examples = 0, 0
    with torch.no_grad():
        for features, targets in data_loader:
            features = features.view(images.shape[0], -1)
            targets = targets
            logits, probs = net(features)
            _, predicted_labels = torch.max(probs, 1)
            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum()
        return correct_pred.float() / num_examples * 100


# Training
start_time = time.time()
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (features, targets) in enumerate(train_loader):
        features = features.view(images.shape[0], -1)

        # forward and backprop
        logits, probs = model(features)
        cost = F.cross_entropy(logits, targets)
        optimizer.zero_grad()

        cost.backward()

        # update params
        optimizer.step()

        if not batch_idx % 200:
            print(
                f"Epoch: {epoch+1:2}/{num_epochs:2}\t"
                + f"Batch: {batch_idx:4}/{len(train_loader):3}\t"
                + f"Cost: {cost:.2}"
            )

    with torch.set_grad_enabled(False):
        print(f"Training Accuracy: {compute_acc(model, train_loader):.3}")

    print(f"Time: {((time.time() - start_time) / 60):.2} min \n")

print(f"Test Accuracy: {compute_acc(model, test_loader):.3}")


# %%
