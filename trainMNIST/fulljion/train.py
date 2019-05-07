# -*- coding: utf-8 -*-
"""
Created on Mon May  6 11:49:28 2019

@author: Yan
"""

# 加载训练需要的模块
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms

# transform模块，主要作用是将输入数据转换成tenssor，并进行归一化
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                              ])

# MNIST数据 train,每一个batch的大小为128
trainset = datasets.MNIST('./MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

# MNIST数据 test,每一个batch的大小为128
testset = datasets.MNIST('./MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True)


# 本次实验所使用的模型，只有全链接层，结构非常简单，没有什么特殊的地方
model = nn.Sequential(nn.Linear(784, 392),
                      nn.ReLU(),
                      nn.Linear(392, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1))

# 定义一下损失函数和优化器
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.003)

# 迭代次数
epochs = 30

# 记录每次的损失
train_losses, test_losses = [], []

# 训练
for e in range(epochs):
    running_loss = 0
    # 读取所有的训练数据，并进行训练
    # images和labels都是维度为128的tensor
    for images, labels in trainloader:
        
        # 将images的size更改一下，变成一条向量
        images = images.view(images.shape[0], -1)
    
        # 清除上一次自动求导的梯度信息
        optimizer.zero_grad()
        
        # forward过程
        output = model(images)
        
        # 计算损失
        loss = criterion(output, labels)
        
        # backward 
        # 此过程中会自动求导
        loss.backward()
        
        # 更新参数
        optimizer.step()
        
        running_loss += loss.item()
    else:
        test_loss = 0
        accuracy = 0
        
        # 利用test数据进行测试
        # 为提高预测的速度，最好关闭梯度计算
        with torch.no_grad():
            for images, labels in testloader:
                
                # 更改images的size
                images = images.view(images.shape[0], -1)
                
                # 预测
                log_ps = model(images)
                
                # 计算损失
                test_loss += criterion(log_ps, labels)
                
                # 由于在pytorch中最终的预测结果都进行了求对数
                # 所以这这里又添加了一个求指数
                ps = torch.exp(log_ps)
                
                # 获取最好的结果
                top_p, top_class = ps.topk(1, dim=1)
                
                # 计算精度
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))
                
        train_losses.append(running_loss/len(trainloader))
        test_losses.append(test_loss/len(testloader))

        # 在每一个epoch中都保存一次模型
        torch.save(model.state_dict(), str(e) +'.pth')

        print("Epoch: {}/{}.. ".format( e+1, epochs),
              "Training Loss: {:.3f}.. ".format(running_loss/len(trainloader)),
              "Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
              "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))
      
# 画一下最终的精度图        
import matplotlib.pyplot as plt
plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Validation loss')
plt.legend(frameon=False)
