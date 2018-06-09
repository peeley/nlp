import torch, torchvision, dataUtils
import torch.nn as nn
import matplotlib.pyplot as plt

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.i2h = nn.Linear(dataUtils.n_letters + 256, 256)
        self.i2o = nn.Linear(dataUtils.n_letters + 256, dataUtils.n_letters)
        self.dropout = nn.Dropout(.1)
        self.softmax = nn.LogSoftmax(dim = 0)
        self.epochs = 1
        self.lr = .0005

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 0)
        this_hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return this_hidden, output

    def train(self, input, target):
        loss_fn = nn.NLLLoss()
        optim = torch.optim.SGD(self.parameters(), lr = self.lr, momentum = .9)
        hidden = torch.zeros(256)
        self.zero_grad()
        loss = 0
        outputString = ''
        for i in range(input.size(0)):
            hidden, output = self.forward(input[i, 0], hidden)
            output = output.view(1, output.size(0))
            loss += loss_fn(output, target[i].view(1,))
            if torch.argmax(output) != len(dataUtils.all_letters) :
                outputString += dataUtils.all_letters[torch.argmax(output)]
        loss.backward()
        optim.step()
        return outputString, (loss / input.size(0))
