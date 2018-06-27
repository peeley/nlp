import torch
import torch.nn as nn

class encoder(nn.Module):
    def __init__(self, inputSize, hiddenSize):
        super(encoder, self).__init__()
        self.hidden_size = hiddenSize
        self.inputSize = inputSize
        self.embedding = nn.Embedding(self.inputSize, self.hiddenSize)
        self.gru = nn.GRU(self.hiddenSize, self.hiddenSize)

    def forward(self, input, hidden):
        embed = self.embedding(input).view(1,1,-1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hiddenSize)

class decoder(nn.Module):
    def __init__(self, outputSize, hiddenSize):
        super(decoder, self).__init__()
        self.hiddenSize = hiddenSize

        self.embed = nn.Embedding(outputSize, hiddenSize)
        self.gru = nn.GRU(self.hiddenSize, self.hiddenSize)
        self.linear = nn.Linear(self.hiddenSize, self.outputSize)
        self.softmax = nn.LogSoftmax(dim = 1)

    def forward(self, input, hidden):
        embed = self.embed(input).view(1,1,-1)
        out = nn.RELU(embed)
        output, hidden = self.gru(out, hidden)
        output = self.linear(output[0])
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hiddenSize)


