import torch
import torch.nn as nn
import numpy

class LSTMnet(nn.Module):
    def __init__(self, vocab = 0,  embedSize = 6, hiddenSize = 6, lr = 1e-3):
        super(LSTMnet, self).__init__()
        self.embedSize = embedSize
        self.hiddenSize = hiddenSize
        self.vocab = vocab
        self.lr = lr

        self.embed = nn.Embedding(self.vocab, self.embedSize)
        self.lstm = nn.LSTM(self.embedSize, self.hiddenSize)
        self.out = nn.Linear(self.hiddenSize, self.vocab)
        self.hidden = self.initHidden()

    def forward(self, input):
        encode = self.embed(input)
        lstmOut, self.hidden = self.lstm(encode.view(len(input), 1, -1), self.hidden)
        out = self.out(lstmOut.view(len(input), -1))
        out = nn.functional.log_softmax( out, dim = 1)
        return out

    def initHidden(self):
        return (torch.zeros(1, 1, self.hiddenSize),
                torch.zeros(1, 1, self.hiddenSize))

    def train(self, input, target):
        self.zero_grad()
        loss_fn = nn.NLLLoss()
        optim = torch.optim.SGD(self.parameters(), lr = self.lr, momentum = .9)
        output = self.forward(input)
        loss = loss_fn(output, target)
        loss.backward()
        optim.step()
        return loss
