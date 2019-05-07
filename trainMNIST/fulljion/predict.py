# -*- coding: utf-8 -*-
"""
Created on Mon May  6 15:54:12 2019

@author: Yan
"""

import torch
from torch import nn
from torchvision import transforms

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                              ])


model = nn.Sequential(nn.Linear(784, 392),
                      nn.ReLU(),
                      nn.Linear(392, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1))



# load model
state_dict = torch.load('29.pth')
model.load_state_dict(state_dict)


import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 打开图片并转成灰度
image = Image.open('8.png')
gray=image.convert('L')

# 显示图片
plt.figure("predict")
plt.imshow(gray)
plt.show()


# 转成tensor
tensor = transform(gray)
tensor = tensor.view(1, 784)
inputdata = torch.autograd.Variable(tensor,requires_grad=False)
outputdata = model(inputdata)
ps = torch.exp(outputdata)

top_p, top_class = ps.topk(1, dim=1)

# 输出结果
print(top_p)
print(top_class)

