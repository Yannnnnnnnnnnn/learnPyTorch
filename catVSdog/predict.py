#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 16:51:36 2019

@author: yqs
"""

import torch
import torch.nn as nn
from torchvision import transforms

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
    transforms.Resize(size=(227,227)),
    transforms.ToTensor(), 
    normalize
])


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




from PIL import Image
import matplotlib.pyplot as plt
import os

f = open('/home/yqs/Desktop/dogs-vs-cats-redux-kernels-edition/predict.txt', 'w')
f.write("id,label \n")
path=r'/home/yqs/Desktop/dogs-vs-cats-redux-kernels-edition/test/'
for root, dirs, files in os.walk(path, topdown=False):
    for name in files:
     
        ## 打开图片并转成灰度
        image = Image.open(os.path.join(root, name))
        # 转成tensor
        tensor = transform(image)
        tensor = tensor.unsqueeze(0)
        inputdata = torch.autograd.Variable(tensor,requires_grad=False)
        outputdata = alexNet(inputdata)
        ps = torch.exp(outputdata)
        
        print("#####################################################")
        
        top_p, top_class = ps.topk(1, dim=1)
        print(top_p)
        print(top_class)

        
        
        filename,extension = os.path.splitext(name)
        # 输出结果
        prob = 0
        if top_class.detach().numpy()[0][0]==0:
            prob = 1 - top_p.detach().numpy()[0][0]
        else:
            prob = top_p.detach().numpy()[0][0]
        print(prob)
        f.write(filename+","+str(prob)+" \n")
f.close()

## 打开图片并转成灰度
#image = Image.open('/home/yqs/Desktop/dogs-vs-cats-redux-kernels-edition/test/6.jpg')
#
## 显示图片
#plt.figure("predict")
#plt.imshow(image)
#plt.show()
#
## 转成tensor
#tensor = transform(image)
#tensor = tensor.unsqueeze(0)
#inputdata = torch.autograd.Variable(tensor,requires_grad=False)
#outputdata = alexNet(inputdata)
#ps = torch.exp(outputdata)
#top_p, top_class = ps.topk(1, dim=1)
## 输出结果
#print(top_p)
#print(top_class)


