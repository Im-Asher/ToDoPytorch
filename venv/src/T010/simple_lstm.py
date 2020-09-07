import torch
import torch.nn as nn
from torch.autograd import Variable

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers =1):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first= True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        x = self.embedding(input)

        output, hidden = self.lstm(x, hidden)
        output = output[:,-1,:]

        output = self.fc(output)
        output = self.softmax(output)

        return output, hidden

    def initHidden(self):
        # 对隐含单元的初始化
        # 注意尺寸是layer_size , batch_size, hidden_size
        # 对隐含单元输出的初始化,全0
        # 注意hidden和cell的维度都是layers,batch_size,hidden_size
        hidden = Variable(torch.zeros(self.num_layers, 1, self.hidden_size))
        # 对隐含单元内部的状态cell的初始化，全0
        cell = Variable(torch.zeros(self.num_layers, 1, self.hidden_size))
        return (hidden, cell)
