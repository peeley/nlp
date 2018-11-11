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
        batchSize= input.shape[1]
        seqLengths = input.shape[0]
        embed = self.embedding(input)
        embed = embed.view(seqLengths, batchSize, self.hiddenSize)
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
        energy = torch.tanh(self.attn(torch.cat([hidden, encoderOutput], 2))) # [B*T*2H]->[B*T*H]
        energy = energy.transpose(2,1) # [B*H*T]
        v = self.v.repeat(encoderOutput.data.shape[0],1).unsqueeze(1) #[B*1*H]
        energy = torch.bmm(v,energy) # [B*1*T]
        return energy.squeeze(1) #[B*T]

    def forward(self, hidden, encoderOutputs):
        seqLengths = encoderOutputs.size(0)
        batchSize = encoderOutputs.size(1)
        H = hidden.repeat(seqLengths, 1, 1).transpose(0,1)
        encoderOutputs = encoderOutputs.transpose(0,1) # [B*T*H]
        attnEnergy = self.score(H, encoderOutputs) # compute attention score
        return nn.functional.softmax(attnEnergy).unsqueeze(1) # normalize with softmax      

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
        self.gru = nn.GRU(hiddenSize * 2, hiddenSize, numLayers, dropout=dropoutProb)
        self.out = nn.Linear(hiddenSize, outputSize)

    def forward(self, input, hidden, encoderOutputs):
        batchSize = input.shape[0]
        embed = self.embed(input).view(1, batchSize, -1) # (1,B,V)
        attnWeights = self.attn(hidden[-1], encoderOutputs)
        context = attnWeights.bmm(encoderOutputs.transpose(0, 1))  # (B,1,V)
        context = context.transpose(0, 1)  # (1,B,V)
        rnnIn = torch.cat((embed, context), 2)
        output, hidden = self.gru(rnnIn, hidden)
        output = output.squeeze(0)  # (1,B,V)->(B,V)
        output = nn.functional.log_softmax(self.out(output))
        return output, hidden

def initHidden(cuda, hiddenSize, layers, batchSize = 1):
    hidden = (torch.zeros(layers, batchSize, hiddenSize), torch.zeros(layers, batchSize, hiddenSize))
    if cuda:
        return (hidden[0].cuda(), hidden[1].cuda())
    return hidden


