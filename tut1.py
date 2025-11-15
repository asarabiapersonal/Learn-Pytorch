import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split

# This function will be used to display an image
def imshow(img, title=None):
    """Imshow for Tensor."""
    # Un-normalize the image
    # The normalization was (input - 0.5) / 0.5
    # So the un-normalization is (input * 0.5) + 0.5
    img = img * 0.5 + 0.5
    
    # Convert from Tensor image
    npimg = img.numpy()
    
    # Reshape the dimensions from (C, H, W) to (H, W, C) for matplotlib
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    
    if title is not None:
        plt.title(title)
    
    plt.pause(0.001) # pause a bit so that plots are updated

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)
print(device," Proceed" if str(device) == "cuda:0" else " CHANGE TO GPU!!!!")


transform_train = transforms.Compose([
    transforms.RandomResizedCrop(size=64, scale=(0.8, 1.0)), # Takes a random crop and resizes to 64x64
    transforms.RandomHorizontalFlip(), # Randomly flips the image horizontally
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
    ])

transform_test = transforms.Compose([
    transforms.Resize(70), # Resize the smaller edge to 70
    transforms.CenterCrop(64), # Crop the center 64x64 pixels
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
    ])

class TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

#create custom class 
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolution layers
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.pool1 = nn.MaxPool2d(2,stride=2)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.pool2 = nn.MaxPool2d(2,stride=2)
        


        # Fully connected layers
        self.fc1 = nn.Linear(25088, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x
 
net = Net()
net = net.to(device)


batch_size =  50
print(f"Batch size set: {batch_size}")

full_dataset = ImageFolder(root="C:/Users/adams/OneDrive/Documents/AI/PetImages")
trainsize = int(4/5 * len(full_dataset))
testsize = len(full_dataset) - trainsize


trainset, testset = random_split(full_dataset, [trainsize, testsize])
print("train test split done")

trainset = TransformedDataset(trainset, transform=transform_train)
testset = TransformedDataset(testset, transform=transform_test)
print("Transforms Applied")

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
print("dataloaders constructed")

# Get one batch of training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# Create a grid from the images and show them


learning_rate=0.01
optimizer = optim.Adam(net.parameters(), lr = learning_rate)
criterion = nn.NLLLoss()

epochs = 10
for epoch in range(epochs):
    running_loss = 0.0000

    for i, data in enumerate(trainloader):
        inputs, labels = data[0].to(device), data[1].to(device)
        if i <= 3:
            plt.figure(figsize=(12, 8)) # Create a new figure to display the images
            img_grid = torchvision.utils.make_grid(inputs.cpu()) # Move to CPU for numpy/matplotlib
            class_names = full_dataset.classes
            imshow(img_grid, title=[class_names[x] for x in labels])
            plt.show() # Use plt.show() to pause execution and see the plot

        optimizer.zero_grad()
        output = net(inputs)
        loss = criterion(output,labels)
        loss.backward()

        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % batch_size == (batch_size - 1):   
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / epochs:.3f}')
            running_loss = 0.0

print('Finished Training')

# prepare to count predictions for each class
class_names = full_dataset.classes
correct_pred = {classname: 0 for classname in class_names}
total_pred = {classname: 0 for classname in class_names}
validText = ""

# again, iterate through test data
with torch.no_grad():
    net.eval()
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[class_names[label]] += 1
                validText = (" Correct Prediction ")
            else:
                validText = (" Incorrect Prediction ")
            #print(validText + " Network Pred: " + str(prediction.item()) + " Actual Class: " + str(label.item()))
            total_pred[class_names[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    print(str(epochs) + " epochs: ")
    if total_pred[classname] != 0:
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
    else:
        print(f'No samples for class: {classname:5s} in the test set')


