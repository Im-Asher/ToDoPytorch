import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision import datasets, models,transforms
import matplotlib.pyplot as plt
import time
import copy
import os

data_dir = './dataset'

image_size = 224
def rightness(predictions, labels):
    '''计算预测错误率的函数，其中predictions是模型给出的一组预测结果，batch_size行num_classes列的矩阵
    ，labels是数据中的正确答案'''
    pred = torch.max(predictions.data, 1)[1]
    rights = pred.eq(labels.data.view_as(pred)).sum()
    return rights, len(labels)
#
train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'),
                                     transforms.Compose([
                                         transforms.RandomSizedCrop(image_size),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])
                                     ]))

val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'),
                                   transforms.Compose([
                                       transforms.Scale(256),
                                       transforms.CenterCrop(image_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                                   ]))

train_loader = torch.utils.data.DataLoader(train_dataset,batch_size = 4, shuffle = True, num_workers = 4)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = 4, shuffle = True, num_workers = 4)

num_classes = len(train_dataset.classes)

# 预训练模式
net = models.resnet18(pretrained=True)

num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, 2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.0001, momentum=0.9)

# 固定值模式
net = models.resnet18()
for param in net.parameters():
    param.requires_grad = False

num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, 2)
criterion = nn.CrossEntropyLoss()
optimizer  = optim.SGD(net.fc.parameters(), lr = 0.001, momentum=0.9)

use_cuda = torch.cuda.is_available()

dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
itype = torch.cuda.LongTensor if use_cuda else torch.LongTensor

net = net.cuda() if use_cuda else net

data, target = Variable(data), Variable(target)

if use_cuda:
    data, target = data.cuda(), target.cuda()

loss = loss.cpu() if use_cuda else loss

# 训练
record = []

num_epochs = 20
net.train(True)
for epoch in range(num_epochs):
    train_rights = []
    train_losses = []
    for batch_idx, (data,target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        output = net(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        right = rightness(output, target)
        train_rights.append(right)
        train_losses.append(loss.data.numpy())
