# -*- coding: utf-8 -*-
"""
Created on Mon May  6 20:47:11 2019

@author: Yan
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May  6 15:42:20 2019

@author: Yan
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May  6 11:49:28 2019

@author: Yan
"""

# Train On MNIST

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms

# define the CNN architecture
class myModel(nn.Module):
    def __init__(self):
        super(myModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, stride=2, padding=1)
        self.fc1   = nn.Linear( 6144,4096)
        self.fc2   = nn.Linear( 4096,4096) 
        self.fc3   = nn.Linear( 4096,10)
        self.dropout = nn.Dropout(0.25)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), 6144 )
        nn.Dropout()
        x = F.relu(self.fc1(x))
        nn.Dropout()
        x = F.relu(self.fc2(x))
        nn.Dropout()
        x = F.relu(self.fc3(x))
        x = self.softmax(x)
        return x

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                              ])

# Download and load the training data
trainset = datasets.MNIST('./MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

# Download and load the test data
testset = datasets.MNIST('./MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True)


model = myModel()

#import numpy as np
#from PIL import Image
#import matplotlib.pyplot as plt
#
## 打开图片并转成灰度
#image = Image.open('8.png')
#gray=image.convert('L')
#
## 显示图片
#plt.figure("predict")
#plt.imshow(gray)
#plt.show()
#
#
## 转成tensor
#tensor = transform(gray)
#tensor = tensor.unsqueeze(0)
#inputdata = torch.autograd.Variable(tensor,requires_grad=False)
#outputdata = model(inputdata)
#
#print(outputdata.shape)

## load model
#state_dict = torch.load('299.pth')
#model.load_state_dict(state_dict)

model.cuda()

criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.003)

epochs = 1000
train_losses, test_losses = [], []

for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # Flatten MNIST images into a 784 long vector
        images = images.cuda()
        labels = labels.cuda()
        
        # TODO: Training pass
        optimizer.zero_grad()
        
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    else:
        test_loss = 0
        accuracy = 0
        
        # Turn off gradients for validation, saves memory and computations
        with torch.no_grad():
            for images, labels in testloader:
                images = images.cuda()
                labels = labels.cuda()
                
                log_ps = model(images)
                test_loss += criterion(log_ps, labels)
                
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))
                
        train_losses.append(running_loss/len(trainloader))
        test_losses.append(test_loss/len(testloader))

        torch.save(model.state_dict(), str(e+1) +'.pth')

        print("Epoch: {}/{}.. ".format( e+1, epochs),
              "Training Loss: {:.3f}.. ".format(running_loss/len(trainloader)),
              "Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
              "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))
      
        
import matplotlib.pyplot as plt
plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Validation loss')
plt.legend(frameon=False)
