import LSTMnet, dataUtils, torch, numpy, sys
import matplotlib.pyplot as plt

sys.settrace
data = dataUtils.constructTweets()['Text']
trainSet = data[:50]
word2idx = dataUtils.word2idx(data)

net = LSTMnet.LSTMnet(vocab = len(word2idx), lr = .05)
length = trainSet.shape[0]
count = 1
epochs = 250

test = True
losses = []

if test == True:
    for j in range(epochs):
        for i in trainSet:
            net.zero_grad()
            net.hidden = net.initHidden()
            loss_fn = torch.nn.NLLLoss()
            optim = torch.optim.SGD(net.parameters(), lr = net.lr, momentum = .9)
            print('Item \t{}/{}\t Epoch #{}'.format(count, length, j+1))
            sentence = i.split()
            if len(sentence) <= 1:
                print('Invalid length, skipping item')
                count += 1
                continue
            input, labels = sentence[:-1], sentence[1:]
            input, labels = dataUtils.prepareSequence(input, word2idx), dataUtils.prepareSequence(labels, word2idx)
            out = (net(input))
            loss = loss_fn(out, labels)
            print('Loss: ', loss.item(), '\n')
            loss.backward()
            optim.step()
            if count % 2 == 0:
                losses.append(loss)
            count += 1
        count = 0

plt.plot(losses)
plt.show()
