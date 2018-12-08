import torch, torchvision, dataUtils
import torch.nn as nn
import matplotlib.pyplot as plt

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.hidden_size = 512
        self.i2h = nn.Linear(dataUtils.n_letters + self.hidden_size, self.hidden_size)
        self.i2o = nn.Linear(dataUtils.n_letters + self.hidden_size, dataUtils.n_letters)
        self.dropout = nn.Dropout(.1)
        self.softmax = nn.LogSoftmax(dim = 0)
        self.epochs = 100
        self.lr = 3e-3

    def forward(self, input, hidden):
        combined = torch.cat((input[0], hidden), 0)
        this_hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return this_hidden, output

    def train(self, input):
        loss_fn = nn.NLLLoss()
        optim = torch.optim.SGD(self.parameters(), lr = self.lr, momentum = .9)
        hidden = torch.rand(self.hidden_size)
        self.zero_grad()
        loss = 0
        outputString = dataUtils.all_letters[torch.argmax(input[0])]
        for i in range(input.size(0)-1):
            hidden, output = self.forward(input[i], hidden)
            output = output.view(1, -1)
            loss += loss_fn(output, torch.argmax(input[i+1]).view(1,))
            if torch.argmax(output) != len(dataUtils.all_letters) :
                outputString += dataUtils.all_letters[torch.argmax(output)]
        loss.backward()
        optim.step()
        return outputString, (loss / input.size(0))
