# %%
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import time

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def plot_train_images(trainloader):
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    imshow(torchvision.utils.make_grid(images))


def compute_accuracy(net, data_loader):
    net.eval()
    correct_pred, num_examples = 0, 0
    with torch.no_grad():
        for images, labels in data_loader:
            images_flatten = images.view(images.shape[0], -1)
            logits, probas = net(images_flatten)
            _, predicted_labels = torch.max(probas, 1)
            num_examples += labels.size(0)
            correct_pred += (predicted_labels == targets).sum()
        
        return correct_pred.float() / num_examples * 100


# %% Load data
transf = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transf
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=4, shuffle=True, num_workers=2
)

testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transf
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=4, shuffle=False, num_workers=2
)

classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

images, labels = iter(trainloader).next()
print(f"Image batch dim: {images.shape} \nLabel dim: {labels.shape}")

# %%

# plot_train_images()

torch.manual_seed(0)
np.random.seed(0)


# Hyperparameters
learning_rate = 0.001
momentum = 0.9
num_epochs = 2

# Architecture
num_features = 3072  # 3x32x32
num_hidden_1 = 128
num_hidden_2 = 256
num_classes = 10


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


model = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

# Training
start_time = time.time()

for epoch in range(num_epochs):
    running_loss = 0.0
    model.train()

    for batch_idx, (images, labels) in enumerate(train_loader):
        images_flatten = images.view(images.shape[0], -1)

        optimizer.zero_grad()

        logits, probas = model(images_flatten)
        loss = criterion(logits, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        if not batch_idx % 50:
            print(f"Epoch: {epoch+1:2}/{num_epochs:2} \tBatch: {batch_idx:3}/{len(trainloader):3} \tLoss: {loss:.2}")




        # if i % 2000 == 1999:
        #     print(f"[{epoch+1},{i+1}] loss: {(running_loss / 2000):.3}")
        #     running_loss = 0.0

print("Finished Training")

# %%

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images_flatten = images.view(images.shape[0], -1)
        _, outputs = net(images_flatten)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

# %%
