## need to build pytorch model
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader


## load dataset
dataset = datasets.SVHN(
    root =  "./data",
    download = True, 
    split = "train", 
    transform =  transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to fit ResNet input size
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize images
    ]))

train_loader = DataLoader(dataset, batch_size = 32, shuffle = True)


## import other models
model  = models.resnet18(pretrained = True)

num_classes = 10 # SVHN has 10 classifications
model.fc = nn.Linear(model.fc.in_features, num_classes)

# freeze layers' parameters
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True

# loss fn and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.fc.parameters(), lr=0.01, momentum=0.9)

# train the model
num_epochs = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.train()

for epoch in range(num_epochs):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

## get results on present data
# evaluation
model.eval()

correct = 0 # correct predictions
total = 0 # totalpredictions

with torch.no_grad():
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted =  torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct/total

print("Accuracy of resnet18 model:", accuracy)