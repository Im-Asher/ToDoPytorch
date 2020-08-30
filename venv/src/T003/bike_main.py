import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt

# %matplotlib inline

data_path = './dataset/hour.csv'
rides = pd.read_csv(data_path)

# 类别数据处理(One-Hot)

dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
for each in dummy_fields:
    dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)
    rides = pd.concat([rides, dummies], axis=1)
fields_to_drop =['instant','dteday','season','weathersit','weekday','atemp','mnth','workingday','hr']
data = rides.drop(fields_to_drop, axis=1)

# 处理数值变量(归一化处理)

quant_features = ['cnt', 'temp', 'hum', 'windspeed']
scaled_features = {}
for each in quant_features:
    mean, std = data[each].mean(), data[each].std()
    scaled_features[each] = [mean, std]
    data.loc[:, each] = (data[each] - mean)/std

# 处理完之后数据集包含17379条记录，59个变量

# 数据集划分

test_data = data[-21*24:]
train_data = data[:-21*24]

target_fields = ['cnt', 'casual', 'registered']

features, targets = train_data.drop(target_fields, axis=1), train_data[target_fields]

test_featrues, test_targets = test_data.drop(target_fields, axis=1), test_data[target_fields]

X = features.values
Y = targets['cnt'].values
Y = Y.astype(float)

Y = np.reshape(Y, [len(Y), 1])
losses = []

input_size = features.shape[1]
hidden_size = 10
output_size = 1
batch_size = 128
neu = torch.nn.Sequential(
    torch.nn.Linear(input_size, hidden_size),
    torch.nn.Sigmoid(),
    torch.nn.Linear(hidden_size, output_size)
)

cost = torch.nn.MSELoss()
optimizer = torch.optim.SGD(neu.parameters(), lr=0.01)

# 模型的训练
for i in range(1000):
    batch_loss = []
    for start in range(0, len(X), batch_size):
        end = start + batch_size if start + batch_size < len(X) else len(X)
        xx = Variable(torch.FloatTensor(X[start:end]))
        yy = Variable(torch.FloatTensor(Y[start:end]))
        predict = neu(xx)
        loss = cost(predict, yy)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_loss.append(loss.data.numpy())
    if i % 100 == 0:
        losses.append(np.mean(batch_loss))
        print(i, np.mean(batch_loss))



targets = test_targets['cnt']
targets = targets.values.reshape([len(targets), 1])
targets = targets.astype(float)

x = Variable(torch.FloatTensor(test_featrues.values))
y = Variable(torch.FloatTensor(targets))


# 用神经网络进行预测

predict = neu(x)

predict = predict.data.numpy()

# 打印输出损失值

# plt.plot(np.arange(len(losses))*100,losses,'o-')
# plt.xlabel('epoch')
# plt.ylabel('MSE')
# plt.show()

# 测试模型的有效性
fig, ax = plt.subplots(figsize = (10, 7))
mean, std = scaled_features['cnt']
ax.plot(predict * std + mean,label='Prediction')
ax.plot(targets * std + mean,label='Data')
ax.legend()
ax.set_xlabel('Data-time')
ax.set_ylabel('Counts')
dates = pd.to_datetime(rides.loc[test_data.index]['dteday'])
dates = dates.apply(lambda d: d.strftime('%b %d'))
ax.set_xticks(np.arange(len(dates))[12::24])
_ = ax.set_xticklabels(dates[12::24], rotation=45)

plt.show()