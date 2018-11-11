import torch, torchtext, pickle
import torch.nn as nn
import torch.nn.utils.rnn as rnn

class encoder(nn.Module):
    def __init__(self, inputSize, hiddenSize = 300,  numLayers = 1):
        super(encoder, self).__init__()
        self.hiddenSize = hiddenSize
        self.inputSize = inputSize
        self.numLayers = numLayers
        self.embedding = nn.Embedding(self.inputSize, self.hiddenSize, padding_idx = 0)
        self.gru = nn.GRU(self.hiddenSize, self.hiddenSize, bidirectional = True, num_layers=numLayers)

    def forward(self, input, hidden):
        batchSize= input.shape[0]
        seqLengths = input.shape[1]
        embed = self.embedding(input)
        embed = embed.view(batchSize, seqLengths, self.hiddenSize)
        output, hidden = self.gru(embed, hidden)
        output = output[:, :, :self.hiddenSize] + output[:, :, self.hiddenSize:]
        return output, hidden

class decoder(nn.Module):
    def __init__(self, outputSize, hiddenSize = 300, dropoutProb = .3, maxLength=10, numLayers = 2):
        super(decoder, self).__init__()
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize
        self.maxLength = maxLength
        self.numLayers = numLayers
        self.dropout = nn.Dropout(dropoutProb)
        self.embed = nn.Embedding(self.outputSize, self.hiddenSize)
        self.lstm = nn.LSTM(self.hiddenSize, self.hiddenSize, num_layers = self.numLayers*2, batch_first = True)
        self.linear = nn.Linear(self.hiddenSize, self.outputSize)
        self.softmax = nn.LogSoftmax(dim = 2)

    def forward(self, input, hidden, out):
        batchSize = input.shape[0]
        embed = self.embed(input).view(batchSize,1, self.hiddenSize)
        out = nn.functional.relu(embed)
        output, hidden = self.lstm(out, hidden)
        output = self.linear(output[:])
        output = self.softmax(output)
        return output, hidden

class attnDecoder(nn.Module):
    def __init__(self, outputSize, hiddenSize = 300, dropoutProb = .3, maxLength = 10, numLayers = 2):
        super(attnDecoder, self).__init__()
        self.maxLength = maxLength 
        self.outputSize = outputSize
        self.hiddenSize = hiddenSize
        self.numLayers = numLayers
        self.embed = nn.Embedding(self.outputSize, self.hiddenSize, padding_idx=0)
        self.dropout = nn.Dropout(dropoutProb)
        self.attn = nn.Linear(self.hiddenSize*2, self.maxLength)
        self.attnCombine = nn.Linear(self.hiddenSize * 3, self.hiddenSize)
        self.lstm = nn.LSTM(self.hiddenSize, self.hiddenSize, num_layers = self.numLayers*2, batch_first = True)
        self.out = nn.Linear(self.hiddenSize, self.outputSize)

    def forward(self, input, hidden, encoderOutputs):
        batchSize = input.shape[0]
        embed = self.embed(input)
        embed = self.dropout(embed)
        hidden = (hidden[0].view(self.numLayers*2, batchSize, self.hiddenSize), 
                hidden[1].view(self.numLayers*2, batchSize, self.hiddenSize))
        lastLayers = (hidden[0][-1] + hidden[0][-2]).view(1, batchSize, self.hiddenSize)
        attn = self.attn(torch.cat((embed, lastLayers), 2))
        attnWeights = nn.functional.softmax(attn, dim=2)
        attnApplied = torch.bmm(attnWeights, encoderOutputs)
        
        output = torch.cat((embed, attnApplied), 2)
        output = self.attnCombine(output)

        output = nn.functional.relu(output)
        output = self.out(output)
        output = nn.functional.log_softmax(output[:], dim=2)
        return output, hidden

class Attn(nn.Module):
    def __init__(self, hiddenSize):
        super(Attn, self).__init__()
        self.hiddenSize = hiddenSize
        self.attn = nn.Linear(self.hiddenSize * 2, hiddenSize)
        self.v = nn.Parameter(torch.FloatTensor(1, hiddenSize))

    def score(self, hidden, encoderOutput):
        energy = self.attn(torch.cat((hidden, encoderOutput[0]), 1))
        energy = self.v.dot(energy[0])
        return energy

    def forward(self, hidden, encoderOutputs):
        maxLen = encoderOutputs.shape[0]
        batchSize = encoderOutputs.shape[1]
        attnEnergies = torch.zeros(batchSize, maxLen)
        for b in range(batchSize):
            for i in range(maxLen):
                attnEnergies[b, i] = self.score(hidden[:, b], encoderOutputs[i, b].unsqueeze(0))
        out = nn.functional.softmax(attnEnergies).unsqueeze(1)
        return out 

class bahdanauDecoder(nn.Module):
    def __init__(self, outputSize, hiddenSize = 300, dropoutProb = .3, maxLength = 10, numLayers = 2):
        super(bahdanauDecoder, self).__init__()
        self.outputSize = outputSize
        self.hiddenSize = hiddenSize
        self.numLayers = numLayers
        self.maxLength = maxLength
        self.embed = nn.Embedding(outputSize, hiddenSize, padding_idx = 0)
        self.dropout = nn.Dropout(dropoutProb)
        self.attn = Attn(self.hiddenSize)
        self.gru = nn.GRU(hiddenSize, hiddenSize, numLayers, dropout=dropoutProb)
        self.concat = nn.Linear(hiddenSize * 2, hiddenSize)
        self.out = nn.Linear(hiddenSize, outputSize)

    def forward(self, input, hidden, encoderOutputs):
        batchSize = input.shape[0]
        embed = self.embed(input)
        embed = self.dropout(embed).view(1,batchSize, self.hiddenSize)
        rnnOut, hidden = self.gru(embed, hidden)
        attnWeights = self.attn(rnnOut, encoderOutputs)
        context = attnWeights.bmm(encoderOutputs.transpose(0,1))
        context = context.transpose(0,1)
        rnnOut = rnnOut.squeeze(0)
        context = context.squeeze(1)
        concatIn = torch.cat((rnnOut, context), 1)
        concatOut = nn.functional.tanh(self.concat(concatIn))
        output = self.out(concatOut)
        return output, hidden, attnWeights

def initHidden(cuda, hiddenSize, layers, batchSize = 1):
    hidden = (torch.zeros(layers, batchSize, hiddenSize), torch.zeros(layers, batchSize, hiddenSize))
    if cuda:
        return (hidden[0].cuda(), hidden[1].cuda())
    return hidden


