import torch
import torch.nn as nn
from torch.autograd import Variable

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size ,output_size, num_layers = 1):
        super(SimpleRNN, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers


        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers, batch_first= True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        x = self.embedding(input)

        output, hidden = self.rnn(x, hidden)

        output = output[:,-1,:]

        output = self.fc(output)

        output = self.softmax(output)

        return output, hidden

    def initHidden(self):
        return Variable(torch.zeros(self.num_layers, 1, self.hidden_size))