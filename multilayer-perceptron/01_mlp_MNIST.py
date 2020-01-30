# %%
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import time

# %%

# Hyperparameters
learning_rate = 0.1
num_epochs = 4
batch_size = 64

# Architecture
num_features = 784
num_hidden_1 = 128
num_hidden_2 = 256
num_classes = 10


# Load data
train_dataset = datasets.MNIST(
    root="data", train=True, transform=transforms.ToTensor(), download=True
)
test_dataset = datasets.MNIST(
    root="data", train=False, transform=transforms.ToTensor(), download=True
)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

images, labels = iter(train_loader).next()
print(f"Image batch dim: {images.shape} \nLabel dim: {labels.shape}")


# Model
class Net(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(Net, self).__init__()

        self.linear_1 = torch.nn.Linear(
            num_features, num_hidden_1
        )  # Xavier/Glorot init
        self.linear_2 = torch.nn.Linear(num_hidden_1, num_hidden_2)
        self.linear_out = torch.nn.Linear(num_hidden_2, num_classes)

    def forward(self, x):
        out = F.relu(self.linear_1(x))
        out = F.relu(self.linear_2(out))
        logits = self.linear_out(out)
        probs = F.log_softmax(logits, dim=1)
        return logits, probs


torch.manual_seed(0)
model = Net(num_features=num_features, num_classes=num_classes)

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# %%
def compute_acc(net, data_loader):
    net.eval()
    correct_pred, num_examples = 0, 0
    with torch.no_grad():
        for features, targets in data_loader:
            features = features.view(-1, 28 * 28)
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
        features = features.view(-1, 28 * 28)

        # forward and backprop
        logits, probs = model(features)
        cost = F.cross_entropy(logits, targets)
        optimizer.zero_grad()

        cost.backward()

        # update params
        optimizer.step()

        if not batch_idx % 50:
            print(
                f"Epoch: {epoch+1:2}/{num_epochs:2}\t"
                + f"Batch: {batch_idx:3}/{len(train_loader):3}\t"
                + f"Cost: {cost:.2}"
            )

    with torch.set_grad_enabled(False):
        print(f"Training Accuracy: {compute_acc(model, train_loader):.3}")

    print(f"Time: {((time.time() - start_time) / 60):.2} min \n")

print(f"Test Accuracy: {compute_acc(model, test_loader):.3}")


# %%
