import torch
import torch.nn as nn
import torch.optim
from torch.autograd import Variable

import re
import jieba
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

good_file = 'dataset/good.txt'
bad_file = 'dataset/bad.txt'

# 将文本中的标点符号过滤掉

def filter_punc(sentence):
 sentence = re.sub("[\s+\.\!\/_,$%^*(+\"\'“”《》?“]+|[+——！，。？、~@#￥%……&*（）：]+", "", sentence)
 return (sentence)

def Prepare_data(good_file, bad_file, is_filter = True):
 all_words = []
 pos_sentences = []
 neg_sentences = []
 with open(good_file, 'r', encoding='utf-8') as fr:
     for idx, line in enumerate(fr):
         if is_filter:
             line = filter_punc(line)
         words = jieba.lcut(line)
         if len(words) > 0:
             all_words += words
             pos_sentences.append(words)
 print('{0} 包含 {1} 行，{2}个词.'.format(good_file, idx+1, len(all_words)))
 count = len(all_words)
 with open(bad_file, 'r', encoding='utf-8') as fr:
     for idx, line in enumerate(fr):
         if is_filter:
             line = filter_punc(line)
             words = jieba.lcut(line)
         if len(words)>0:
             all_words += words
             neg_sentences.append(words)
 print('{0} 包含 {1} 行，{2}个词.'.format(bad_file, idx + 1, len(all_words)-count))

 diction = {}
 cnt = Counter(all_words)
 for word, freq in cnt.items():
     diction[word] = [len(diction), freq]
 print('字典大小:{}'.format(len(diction)))
 return (pos_sentences, neg_sentences, diction)

pos_sentences, neg_sentences, diction = Prepare_data(good_file, bad_file, True)
st = sorted([(v[1], w) for w,v in diction.items()])

def word2index(word, diction):
 if word in diction:
     value = diction[word][0]
 else:
     value = -1
 return(value)


def index2word(index, dicition):
    for w,v in diction.items():
        if v[0] == index:
            return (w)
    return (None)

# 输入一个句子和相应的词典，得到这个句子的向量化表示
# 向量的尺寸为词典中词汇的个数，i位置上面的数值为第i个单词在sentence中出现的频率

def sentence2vec(sentence, dicitionary):
    vector = np.zeros(len(dicitionary))
    for l in sentence:
        vector[l] += 1
    return (1.0 * vector / len(sentence))

# 遍历所有句子，将每一个词映射成编码

dataset = []
labels = []
sentences = []
# 处理正向评论
for sentence in pos_sentences:
    new_sentence = []
    for l in sentence:
        if l in diction:
            new_sentence.append(word2index(l, diction))
    dataset.append(sentence2vec(new_sentence, diction))
    labels.append(0)
    sentences.append(sentence)

# 处理负向评论
for sentence in neg_sentences:
    new_sentence = []
    for l in sentence:
        if l in diction:
            new_sentence.append(word2index(l, diction))
    dataset.append(sentence2vec(new_sentence, diction))
    labels.append(0)
    sentences.append(sentence)

# 打乱所有的数据顺序，形成数据集
# indices 为所有数据下标的全排序

indices = np.random.permutation(len(dataset))

# 根据打乱的下标，重新生成数据集dataset、标签机labels，以及对应的原始句子sentences
dataset = [dataset[i] for i in indices]
labels = [labels[i] for i in indices]
sentences = [sentences[i] for i in indices]


# 数据集划分 train 训练集 valid 校验集 test 测试集

test_size = len(dataset) // 10
train_data = dataset[2 * test_size :]
train_label = labels[2 * test_size :]

valid_data = dataset[: test_size]
valid_label = labels[: test_size]

test_data = dataset[test_size : 2 * test_size]
test_label = labels[test_size : 2 * test_size]



# 神经网络的构建
# 一个简单前馈神经网络，共3层
# 第一层为线性层，加一个非线性ReLu，第二层为线性层，中间由10个隐含层神经元
# 输出维度为词典的大小，每一段评论的词袋模型
model = nn.Sequential(
    nn.Linear(len(diction), 10),
    nn.ReLU(),
    nn.Linear(10, 2),
    nn.LogSoftmax(dim=1),
)

# 自定义一组数据分类准确度的函数
# predictions 为模型给出的预测结果，labels为数据中的标签。比较二者以确定整个神经网路当前的表现
def rightness(predictions, labels):
    '''计算预测错误率的函数，其中predictions是模型给出的一组预测结果，batch_size行num_classes列的矩阵
    ，labels是数据中的正确答案'''
    pred = torch.max(predictions.data, 1)[1]
    rights = pred.eq(labels.data.view_as(pred)).sum()
    return rights, len(labels)



# 模型训练
cost = torch.nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
records = []

losses = []
for epoch in range(10):
    for i , data in enumerate(zip(train_data, train_label)):
        x, y = data
        if torch.cuda.is_available():
            x = Variable(torch.FloatTensor(x).view(1, -1)).cuda()
            y = Variable(torch.LongTensor(np.array([y]))).cuda()
            model.cuda()
        else:
            x = Variable(torch.FloatTensor(x).view(1, -1))
            y = Variable(torch.LongTensor(np.array([y])))

        optimizer.zero_grad()
        predict = model(x)
        loss = cost(predict, y)
        losses.append(loss.data.cpu().numpy())
        loss.backward()
        optimizer.step()

        # 每3000次使用校验集上的数据进行校验
        if i % 3000 == 0:
            val_losses = []
            rights = []
            for j, val in enumerate(zip(valid_data, valid_label)):
                x, y = val
                if torch.cuda.is_available():
                    x = Variable(torch.FloatTensor(x).view(1, -1)).cuda()
                    y = Variable(torch.LongTensor(np.array([y]))).cuda()
                else:
                    x = Variable(torch.FloatTensor(x).view(1, -1))
                    y = Variable(torch.LongTensor(np.array([y])))
                predict = model(x)

                right = rightness(predict, y)
                rights.append(right)
                loss = cost(predict, y)
                val_losses.append(loss.data.cpu().numpy())

            right_ratio = 1.0 * np.sum([i[0] for i in rights]) / np.sum([i[1] for i in rights])
            print('第{}轮，训练损失：{:.2f}，检验损失：{:.2f}，校验准确率：{:.2f}'.format(epoch,
                                                                     np.mean(losses), np.mean(val_losses), right_ratio))
            records.append([np.mean(losses),np.mean(val_losses),right_ratio])




# 绘制误差曲线
# a = [i[0] for i in records]
# b = [i[1] for i in records]
# c = [i[2] for i in records]
# plt.plot(a, label = 'Train Loss')
# plt.plot(b, label = 'Valid Loss')
# plt.plot(c, label = 'Valid Accuracy')
# plt.xlabel('Steps')
# plt.ylabel('Loss & Accuracy')
# plt.legend()
# plt.show()

# torch.save(model,'save_model/model.mdl')
# model = torch.load('save_model/model.mdl')

vals = []
for data ,target in zip(test_data, test_label):
    if torch.cuda.is_available():
        data, target = Variable(torch.FloatTensor(data).view(1, -1)).cuda(), \
                       Variable(torch.LongTensor(np.array([target]))).cuda()
    else:
        data, target = Variable(torch.FloatTensor(data).view(1, -1)),\
            Variable(torch.LongTensor(np.array([target])))
    output = model(data)
    val = rightness(output, target)
    vals.append(val)

rights = (sum([tup[0] for tup in vals])), sum([tup[1] for tup in vals])
right_rate = 1.0 * rights[0] / rights[1]

print(right_rate)


