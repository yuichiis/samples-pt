import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

batch_size = 128

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, ), (0.5, ))])
trainset = torchvision.datasets.MNIST(root='./data',
                                        train=True,
                                        download=True,
                                        transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,
                                            batch_size=batch_size,
                                            shuffle=True,)
                                            #num_workers=2)

testset = torchvision.datasets.MNIST(root='./data',
                                        train=False,
                                        download=True,
                                        transform=transform)
testloader = torch.utils.data.DataLoader(testset,
                                            batch_size=batch_size,
                                            shuffle=False,)
                                            #num_workers=2)

classes = tuple(np.linspace(0, 9, 10, dtype=np.uint8))
train_size = len(trainset)
print(trainset)
print(len(trainset))
image,label = trainset[0]
print(image.shape)
print(label)

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3) # 28x28x32 -> 26x26x32
        self.conv2 = nn.Conv2d(32, 64, 3) # 26x26x64 -> 24x24x64
        self.pool = nn.MaxPool2d(2, 2) # 24x24x64 -> 12x12x64
        self.dropout1 = nn.Dropout2d()
        self.fc1 = nn.Linear(12 * 12 * 64, 128)
        self.dropout2 = nn.Dropout2d()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = x.view(-1, 12 * 12 * 64)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.layers(x)

#from net import Net

model = Net()
epochs = 5

criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters())

def progress(epoch,num,running_loss):
    print('\rEpoch {:d}, {:5d} => loss: {:.3f}'
        .format(epoch + 1, num, running_loss) ,end="")

print('Start Training')

for epoch in range(epochs):
    running_loss = 0.0
    epoch_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader, 0):
        # zero the parameter gradients

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:
            progress(epoch,(i + 1)*batch_size,running_loss/(i + 1))
    progress(epoch,train_size,running_loss/i)
    print()

print('Finished Training')

correct = 0
total = 0

with torch.no_grad():
    for (images, labels) in testloader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy: {:.2f} %%'.format(100 * float(correct/total)))
