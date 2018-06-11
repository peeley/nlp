import LSTMnet, dataUtils, torch
import numpy

data = dataUtils.constructJoke()
trainSet = data[:5000]
testSet = data[: -int((trainSet.shape[0] / 500))]
word2idx = dataUtils.word2idx(data)

net = LSTMnet.LSTMnet(vocab = len(word2idx), lr = .1)
print(word2idx)

for i in trainSet:
    sentence = i.split()
    if len(sentence) <= 1:
        print(i)
        continue
    input, labels = sentence[:-1], sentence[1:]
    input, labels = dataUtils.prepareSequence(input, word2idx), dataUtils.prepareSequence(labels, word2idx)
    net.train(input, labels)
