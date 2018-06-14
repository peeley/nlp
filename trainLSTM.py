
import LSTMnet, dataUtils, torch, numpy, sys

sys.settrace
data = dataUtils.constructJoke()
trainSet = data[:]
testSet = data[: -int((trainSet.shape[0] / 500))]
word2idx = dataUtils.word2idx(data)

net = LSTMnet.LSTMnet(vocab = len(word2idx), lr = .1)
print(word2idx)
length = trainSet.shape[0]
count = 1
for i in trainSet:
    try:
        print('Importing raw sentence:', i)
        sentence = i.split()
        print('Splitting sentence')
        if len(sentence) <= 1:
            print('SHORT SENTENCE IN DATASET')
            continue
        input, labels = sentence[:-1], sentence[1:]
        input, labels = dataUtils.prepareSequence(input, word2idx), dataUtils.prepareSequence(labels, word2idx)
        print('Training item')
        net.train(input, labels)
        print('Moving to next item, {}/{}\n'.format(count, length))
        count += 1
    except:
        print('SOMETHING GOOFED BADLY')
