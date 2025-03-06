import torch
import torch.nn as nn
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import tqdm
import matplotlib.pyplot as plt

BATCH_SIZE = 32
INPUT_SIZE = 28 * 28
OUTPUT_SIZE = 10
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loads in the MNIST dataset and create dataloaders for them
train_data = torchvision.datasets.MNIST("./", train=True, transform=transforms.ToTensor(), download=True)
test_data = torchvision.datasets.MNIST("./", train=False, transform=transforms.ToTensor(), download=True)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)

# Trains the model for one epoch
def train(model, optimizer, criterion, epoch):
    model.train()
    for image, label in tqdm.tqdm(train_loader, desc=f"Epoch {epoch}"):
        image = image.to(DEVICE)
        label = label.to(DEVICE)

        logits = model(image).to(DEVICE)
        loss = criterion(logits, label).to(DEVICE)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Goes through the validation dataset and find the accuracy and average loss
def validate(model, optimizer, criterion):
    model.eval()
    size = len(test_loader.dataset)
    num_batches = len(test_loader)
    average_loss, correct = 0, 0

    for image, label in test_loader:
        image = image.to(DEVICE)
        label = label.to(DEVICE)

        logits = model(image).to(DEVICE)
        loss = criterion(logits, label).to(DEVICE)

        average_loss += loss.item()
        correct += (logits.argmax(1) == label).sum().item()
    
    average_loss /= num_batches
    correct /= size
    print(f"Accuracy: {(100*correct):>0.1f}%, Avg loss: {average_loss:>8f} \n")

# Creates the Neural Network  
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(INPUT_SIZE, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, OUTPUT_SIZE),
    nn.Softmax(dim=-1),
).to(DEVICE)

criterion = nn.CrossEntropyLoss().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Shows the performance and accuracy of the network at the start
print("Starting Performance: ")
validate(model, optimizer, criterion)

# Trains the network
for epoch in range(1, EPOCHS + 1):
    train(model, optimizer, criterion, epoch)
    validate(model, optimizer, criterion)

# Show 10 examples from the validation dataset in MatPlotLib 
num_rows = 2
num_cols = 5
figure, axis = plt.subplots(num_rows, num_cols)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)

for i in range(num_rows):
    for j in range(num_cols):
        image, label = next(iter(test_loader))
        image = image.to(DEVICE)
        label = label.to(DEVICE)

        predicted = model(image).argmax(dim=1).item()
        axis[i, j].set_title(f"Label: {predicted}")
        axis[i, j].imshow(image.reshape(28, 28).cpu())
        axis[i, j].axis('off')

plt.show()
