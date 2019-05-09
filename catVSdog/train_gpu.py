# -*- coding: utf-8 -*-
"""
Created on Tue May  7 14:06:58 2019

@author: Yan
"""

import torch
import torch.nn as nn
from torch import optim
from torchvision.datasets import ImageFolder
from torchvision import transforms

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
    transforms.Resize(size=(227,227)),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), #将图片转换为Tensor,归一化至[0,1]
    normalize
])

train_dataset = ImageFolder(r'/home/yqs/Desktop/dogs-vs-cats-redux-kernels-edition/train',transform=transform)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=512, shuffle=True)

validation_dataset = ImageFolder(r'/home/yqs/Desktop/dogs-vs-cats-redux-kernels-edition/validation',transform=transform)
validationloader = torch.utils.data.DataLoader(validation_dataset, batch_size=512, shuffle=True)

# AlexNet
class MineAlexNet(nn.Module):
    def __init__(self, num_classes=2):
        super(MineAlexNet, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        x = self.logsoftmax(x)
        return x
    

# load pretrained
state_dict = torch.load('37.pth')
alexNet = MineAlexNet(2)
alexNet.load_state_dict(state_dict)


print(alexNet)

# cuda
alexNet.cuda()
print("cuda...")

# images,labels = next(iter(validationloader))
# inputdata = torch.autograd.Variable(images[0].unsqueeze(0),requires_grad=False)
# outputdata = alexNet(inputdata)
# print(outputdata)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(alexNet.parameters(), lr=0.00005)

epochs = 20
train_losses, validation_losses = [], []

for e in range(epochs):
    running_loss = 0
    for images,labels in trainloader:
        
        images = images.cuda()
        labels = labels.cuda()
       
        # TODO: Training pass
        optimizer.zero_grad()
        
        output = alexNet(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    else:
        validation_loss = 0
        accuracy = 0
        
        # Turn off gradients for validation, saves memory and computations
        with torch.no_grad():
            for images, labels in validationloader:
                images = images.cuda()
                labels = labels.cuda()
                
                log_ps = alexNet(images)
                validation_loss += criterion(log_ps, labels)
                
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))
                
        train_losses.append(running_loss/len(trainloader))
        validation_losses.append(validation_loss/len(validationloader))

        torch.save(alexNet.state_dict(), str(e+1+37) +'.pth')

        print("Epoch: {}/{}.. ".format( e+1, epochs),
              "Training Loss: {:.3f}.. ".format(running_loss/len(trainloader)),
              "Test Loss: {:.3f}.. ".format(validation_loss/len(validationloader)),
              "Test Accuracy: {:.3f}".format(accuracy/len(validationloader)))
      
        
import matplotlib.pyplot as plt
plt.plot(train_losses, label='Training loss')
plt.plot(validation_losses, label='Validation loss')
plt.legend(frameon=False)


