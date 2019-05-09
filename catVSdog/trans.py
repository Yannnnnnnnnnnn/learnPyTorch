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

transform = transforms.Compose([
    transforms.Resize(227),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), #将图片转换为Tensor,归一化至[0,1]
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
state_dict = torch.load('alexnetImageNet.pth')
alexNet = MineAlexNet(1000)
alexNet.load_state_dict(state_dict)

# update num_classes
alexNet.classifier[6] = nn.Linear(4096, 2)


print(alexNet)

torch.save(alexNet.state_dict(), 'begin.pth')

# cuda
alexNet.cuda()
print("cuda...")



