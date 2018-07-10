import torch, torchtext, pickle
import torch.nn as nn

class encoder(nn.Module):
    def __init__(self, inputSize, hiddenSize = 300, lr = 1e-3):
        super(encoder, self).__init__()
        self.hiddenSize = hiddenSize
        self.inputSize = inputSize
        #self.glove = pickle.load(open('data/embeddings/english/glove.pkl', 'rb'))
        #self.embedding = nn.Embedding.from_pretrained(self.glove, freeze = True)
        self.embedding = nn.Embedding(self.inputSize, self.hiddenSize)
        #self.gru = nn.GRU(self.hiddenSize, self.hiddenSize)
        self.lstm = nn.LSTM(self.hiddenSize, self.hiddenSize)
        self.lr = lr

    def forward(self, input, hidden):
        embed = self.embedding(input)
        embed = embed.view(1, 1, -1)
        #output, hidden = self.gru(embed, hidden)
        output, hidden = self.lstm(embed, hidden)
        return output, hidden

    def initHidden(self):
        return (torch.zeros(1, 1, self.hiddenSize), torch.zeros(1, 1, self.hiddenSize))

class decoder(nn.Module):
    def __init__(self, outputSize, hiddenSize = 300, lr = 1e-3, dropoutProb = .1):
        super(decoder, self).__init__()
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize
        self.dropout = nn.Dropout(dropoutProb)
        self.embed = nn.Embedding(self.outputSize, self.hiddenSize)
        #self.gru = nn.GRU(self.hiddenSize, self.hiddenSize)
        self.lstm = nn.LSTM(self.hiddenSize, self.hiddenSize)
        self.linear = nn.Linear(self.hiddenSize, self.outputSize)
        self.softmax = nn.LogSoftmax(dim = 1)
        self.lr = lr

    def forward(self, input, hidden):
        embed = self.embed(input).view(1,1,-1)
        out = nn.functional.relu(embed)
        #output, hidden = self.gru(out, hidden)
        output, hidden = self.lstm(out, hidden)
        output = self.linear(output[0])
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return (torch.zeros(1, 1, self.hiddenSize), torch.zeros(1,1, self.hiddenSize))


