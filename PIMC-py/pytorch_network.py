import torch
from torch import nn
import matplotlib.pyplot as plt

# Creating a PyTorch Dataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from Data import train_loader, test_loader



num_epochs = 10000

learning_rate = 10**(-3)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers_stack = nn.Sequential(
            nn.Linear(2, 5),
            nn.ReLU(),
            nn.Linear(5, 5),
            nn.Sigmoid(),
            nn.Linear(5, 1),
        )

    def forward(self, x):
        logits = self.layers_stack(x)
        return logits
    
    



    
    
model = NeuralNetwork()
#print(model)    


criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

def train_loop(dataloader, model, criterion, optimizer):
    num_batches = len(dataloader)

    train_loss = 0

    for features, targets in dataloader:
        # Compute prediction and loss
        pred = model(features)
        loss = criterion(pred, targets)

        # Optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= num_batches
    print(f"Train loss: {train_loss:>8f}")

    return train_loss


def test_loop(dataloader, model, criterion):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    test_loss = 0

    with torch.no_grad():
        for features, targets in dataloader:
            # Compute prediction and loss
            pred = model(features)
            loss = criterion(pred, targets)

            test_loss += loss.item()


    test_loss /= num_batches
    print(f"Test loss: {test_loss:>8f} \n")

    return test_loss








loss_history = {"train": [], "test": []}

for i in range(num_epochs):
    print(f"Epoch {i+1}")
    train_loss = train_loop(train_loader, model, criterion, optimizer)
    test_loss = test_loop(test_loader, model, criterion)

    loss_history["train"].append(train_loss)
    loss_history["test"].append(test_loss)
print("Done!")

Epochs=[i for i in range(num_epochs)]
plt.figure()
plt.title("learning_rate="+str(learning_rate),fontsize=17)
plt.plot(Epochs,loss_history["train"])
plt.plot(Epochs,loss_history["test"])
plt.legend(["train","test"])
plt.ylabel("MSE",fontsize=17)
plt.xlabel("epoch",fontsize=17)
plt.grid(True)
plt.show()

torch.save(model.state_dict(), "model_weights1.pth")
