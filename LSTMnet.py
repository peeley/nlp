import torch
import torch.nn as nn
import numpy

class LSTMnet(nn.Module):
    def __init__(self):
        super(LSTMnet, self).__init__()
        self.embedSize = 10
        self.hiddenSize = 6
        self.vocab = 1000
        self.lr = 1e-3

        self.embed = nn.Embedding(self.vocab, self.embedSize)
        self.lstm = nn.LSTM(self.embedSize, self.hiddenSize)
        self.out = nn.Linear(self.hiddenSize, self.vocab)
        self.hidden = self.initHidden()

    def forward(self, input):
        encode = self.embed(input)
        lstmOut, self.hidden = self.LSTM(len(input), 1, -1, self.hidden)
        out = self.out(lstmOut.view(len(input), -1))
        return out

    def initHidden(self):
        return (torch.zeros(1, 1, self.hiddenSize),
                torch.zeros(1, 1, self.hiddenSize))

    def train(self, input, target):
        loss_fn = nn.CrossEntropyLoss()
        optim = nn.optim.Adam(self.parameters(), lr = self.lr)
        # TODO : finish train loop

