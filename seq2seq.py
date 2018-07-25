import torch, torchtext, pickle
import torch.nn as nn

class encoder(nn.Module):
    def __init__(self, inputSize, hiddenSize = 300, lr = 1e-3, numLayers = 1):
        super(encoder, self).__init__()
        self.hiddenSize = hiddenSize
        self.inputSize = inputSize
        self.numLayers = numLayers
        self.embedding = nn.Embedding(self.inputSize, self.hiddenSize)
        self.lstm = nn.LSTM(self.hiddenSize, self.hiddenSize, bidirectional = True, num_layers=numLayers)
        self.lr = lr

    def forward(self, input, hidden):
        embed = self.embedding(input)
        embed = embed.view(1, 1, -1)
        output, hidden = self.lstm(embed, hidden)
        return output, hidden

class decoder(nn.Module):
    def __init__(self, outputSize, hiddenSize = 300, lr = 1e-3, dropoutProb = .1, maxLength=10):
        super(decoder, self).__init__()
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize
        self.maxLength = maxLength
        self.dropout = nn.Dropout(dropoutProb)
        self.embed = nn.Embedding(self.outputSize, self.hiddenSize)
        self.lstm = nn.LSTM(self.hiddenSize, self.hiddenSize)
        self.linear = nn.Linear(self.hiddenSize, self.outputSize)
        self.softmax = nn.LogSoftmax(dim = 1)
        self.lr = lr

    def forward(self, input, hidden, out):
        embed = self.embed(input).view(1,1,-1)
        out = nn.functional.relu(embed)
        output, hidden = self.lstm(out, hidden)
        output = self.linear(output[0])
        output = self.softmax(output)
        return output, hidden

class attnDecoder(nn.Module):
    def __init__(self, outputSize, hiddenSize = 300, lr = 1e-3, dropoutProb = .1, maxLength = 10, numLayers = 2):
        super(attnDecoder, self).__init__()
        self.lr = lr
        self.maxLength = maxLength 
        self.outputSize = outputSize
        self.hiddenSize = hiddenSize
        self.numLayers = numLayers
        self.embed = nn.Embedding(self.outputSize, self.hiddenSize)
        self.dropout = nn.Dropout(dropoutProb)
        self.attn = nn.Linear(self.hiddenSize * 2, self.maxLength)
        # bidirectional encoding requires input of attnCombine = hiddenSize * 3 for some reason
        # but unidirectional allows hiddenSize * 2
        self.attnCombine = nn.Linear(self.hiddenSize * 3, self.hiddenSize)
        self.lstm = nn.LSTM(self.hiddenSize, self.hiddenSize, num_layers = self.numLayers)
        self.out = nn.Linear(self.hiddenSize, self.outputSize)

    def forward(self, input, hidden, encoderOutputs):
        embed = self.embed(input).view(1,1,-1)
        embed = self.dropout(embed)
        attn = self.attn(torch.cat((embed[0], hidden[0][0]), 1))
        attnWeights = nn.functional.softmax(attn, dim=1)
        attnApplied = torch.bmm(attnWeights.unsqueeze(0), encoderOutputs.unsqueeze(0))
        
        output = torch.cat((embed[0], attnApplied[0]), 1)
        output = self.attnCombine(output).unsqueeze(0)

        output = nn.functional.relu(output)
        output, hidden = self.lstm(output, hidden)
        output = nn.functional.log_softmax(self.out(output[0]), dim=1)
        return output, hidden

def initHidden(cuda, hiddenSize, layers):
    if cuda:
        return (torch.zeros(layers, 1, hiddenSize).cuda(), torch.zeros(layers, 1, hiddenSize).cuda())
    return (torch.zeros(layers, 1, hiddenSize), torch.zeros(layers, 1, hiddenSize))



