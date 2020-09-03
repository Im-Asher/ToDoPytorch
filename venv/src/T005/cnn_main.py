import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

import torchvision.datasets as dsets
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np


# 定义超参数

image_size = 28
num_classes = 10
num_epochs = 20
batch_size = 64

def rightness(predictions, labels):
    '''计算预测错误率的函数，其中predictions是模型给出的一组预测结果，batch_size行num_classes列的矩阵
    ，labels是数据中的正确答案'''
    pred = torch.max(predictions.data, 1)[1]
    rights = pred.eq(labels.data.view_as(pred)).sum()
    return rights, len(labels)
# 下载数据集

train_dataset = dsets.MNIST(root='./dataset',
                            train=True,
                            transform = transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='./dataset',
                           train=False,
                           transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle = True)
'''我们希望将测试数据分成两部分，一部分作为校验数据，一部分作为测试数据。
校验数据用于检测模型是否过拟合并调整参数，测试数据检验整个模型的工作'''

# 首先，定义下标数组indices，它相当于对所有test_dataset中数据的编码
# 然后，定义下标indices_val表示校验集数据的下标，indices_test表示测试集下标
indices = range(len(test_dataset))
indices_val = indices[:5000]
indices_test = indices[5000:]



sampler_val = torch.utils.data.sampler.SubsetRandomSampler(indices_val)
sampler_test = torch.utils.data.sampler.SubsetRandomSampler(indices_test)

validation_loader = torch.utils.data.DataLoader(
    dataset = test_dataset,
    batch_size = batch_size,
    shuffle = False,
    sampler = sampler_val
)

test_loader = torch.utils.data.DataLoader(
    dataset = test_dataset,
    batch_size = batch_size,
    shuffle = False,
    sampler = sampler_test
)

# 测试数据集
# 随便从数据集中读入一张图片，并绘制出来

idx = 133
# dataset支持下标索引，其中提取出来的元素为feature、target格式，即属性和标签。[0]表示索引features
muteimg = train_dataset[idx][0].numpy()

# 一般的图像包含RGB这3个通道，而MNIST数据集的图像都是灰度的，只有一个通道
# 因此，我们忽略通道，把图像看作一个灰度矩阵
# 用imshow画图，会将灰度矩阵自动展现为彩色，不同灰度对应不同的颜色：从黄到紫

# plt.imshow(muteimg[0,...])
# print('标签是：',train_dataset[idx][1])
# plt.show()

# 构建网络
depth = [4, 8]

class ConvNet(nn.Module):
    def __init__(self):
        # 该函数在创建一个ConvNet对象即调用语句net=ConvNet()时就会被调用
        # 首先调用父类相应的构造函数
        super(ConvNet, self).__init__()

        # 其次构造ConvNet需要用到的各个神经模块
        # 注意，定义组件并不是真正搭建组件，只是把基本建筑砖块先找好
        # 定义一个卷积层，输入通道为1，输出通道为4，窗口大小为5，padding=2
        self.conv1 = nn.Conv2d(1, 4, 5, padding= 2)
        self.pool = nn.MaxPool2d(2,2) #定义一个池化层，一个窗口为2X2的池化运算
        # 第二层卷积，输入通道为depth[0]，输出通道为depth[1]，窗口为5，padding=2
        self.conv2 = nn.Conv2d(depth[0], depth[1], 5, padding= 2)
        # 一个线性连接层，输入尺寸为最后一层立方体的线性平铺，输出层512个节点。
        self.fc1 = nn.Linear(image_size // 4 * image_size // 4 * depth[1], 512)

        self.fc2 = nn.Linear(512, num_classes)#最后一层线性分类单元，输入为512，输出为要做分类的类别数

    def forward(self, x):
        x = self.conv1(x)

        x = F.relu(x)

        x = self.pool(x)

        x = self.conv2(x)

        x = F.relu(x)

        x = self.pool(x)

        x = x.view(-1, image_size // 4 * image_size // 4 * depth[1])

        x = F.relu(self.fc1(x))

        x = F.dropout(x, training=self.training)

        x = self.fc2(x)

        x = F.log_softmax(x, dim=1)

        return x

    def retrieve_features(self, x):
        feature_map1 = F.relu(self.conv1(x))
        x = self.pool(feature_map1)
        feature_map2 = F.relu(self.conv2(x))
        return (feature_map1,feature_map2)

# 运行模型
if torch.cuda.is_available():
    net = ConvNet().cuda()
else:
    net =  ConvNet()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

record = []
weights = []

# for epoch in range(num_epochs):
#
#     train_rights = []
#
#     for batch_idx, (data, target) in enumerate(train_loader):
#         if torch.cuda.is_available():
#             data, target = Variable(data).cuda(), Variable(target).cuda()
#         else:
#             data, target = Variable(data), Variable(target)
#         net.train()
#         output = net(data)
#         loss = criterion(output, target)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         right = rightness(output, target)
#         train_rights.append(right)
#
#
#         if batch_idx % 100 == 0:
#             net.eval()
#             val_rights = []
#
#             for(data, target) in validation_loader:
#                 if torch.cuda.is_available():
#                     data, target = Variable(data).cuda(), Variable(target).cuda()
#                 else:
#                     data, target = Variable(data), Variable(target)
#                 output = net(data)
#                 right = rightness(output, target)
#                 val_rights.append(right)
#
#             train_r = (sum([tup[0] for tup in train_rights]), sum([tup[1] for tup in train_rights]))
#             val_r = (sum([tup[0] for tup in val_rights]), sum([tup[1] for tup in val_rights]))
#             print('训练周期:{} [{}/{} ({:.0f}%)]\t, loss: {:.6f}\t, 训练正确率: {:.2f}%\t,'
#                   '校验正确率：{:.2f}%'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader), loss.item(),
#                 100. * train_r[0] / train_r[1],
#                 100. * val_r[0] / val_r[1]
#             ))
#
#             record.append((100 - 100. * train_r[0] / train_r[1], 100 - 100. * val_r[0]/val_r[1]))
#
#             weights.append([net.conv1.weight.data.clone(), net.conv1.bias.data.clone(),
#                             net.conv2.weight.data.clone(), net.conv2.bias.data.clone()])


# 测试模型
# torch.save(net,'./save_model/cnn_model.mdl')
net = torch.load('./save_model/cnn_model.mdl')
# print('保存完毕')
net.eval()
vals = []

for data, target in test_loader:
    if torch.cuda.is_available():
        data, target = Variable(data).cuda(), Variable(target).cuda()
    else:
        data, target = Variable(data, volatile = True), Variable(target)
    output = net(data)
    val = rightness(output, target)
    vals.append(val)

rights = (sum([tup[0] for tup in vals]), sum([tup[1] for tup in vals]))
right_rate = 1.0 * rights[0] / rights[1]
right_rate


# 绘制训练过程的误差曲线，校验集和测试集上的错误率

plt.figure(figsize=(10, 7))
plt.plot(record)
plt.xlabel('Steps')
plt.ylabel('Error rate')
plt.show()