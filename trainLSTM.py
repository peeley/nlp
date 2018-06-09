import LSTMnet, dataUtils, torch
import numpy

data = dataUtils.constructJoke()
trainSet = data[:5000]
testSet = data[: -int((trainSet.shape[0] / 500))]

net = LSTMnet.LSTMnet()

for i in trainSet:
    sentence = i.split()
    input, labels = sentence[:-1], sentence[1:]
    # TODO: represent input as one-hot vectors in vocab 
    net.train(input, labels)
