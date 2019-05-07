# -*- coding: utf-8 -*-
"""
Created on Mon May  6 15:54:12 2019

@author: Yan
"""

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

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                              ])


model = myModel()



# load model
state_dict = torch.load('53.pth')
model.load_state_dict(state_dict)


import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 打开图片并转成灰度
image = Image.open('9_M.png')
gray=image.convert('L')

# 显示图片
plt.figure("predict")
plt.imshow(gray)
plt.show()


# 转成tensor
tensor = transform(gray)
tensor = tensor.unsqueeze(0)
inputdata = torch.autograd.Variable(tensor,requires_grad=False)
outputdata = model(inputdata)
ps = torch.exp(outputdata)

top_p, top_class = ps.topk(1, dim=1)

# 输出结果
print(top_p)
print(top_class)

