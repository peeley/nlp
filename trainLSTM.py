import LSTMnet, dataUtils, torch, numpy, sys
sys.settrace
data = dataUtils.constructTweets()
trainSet = data['Text']
testSet = data[: -int(trainSet.shape[0] / 500)]
word2idx = dataUtils.word2idx(data['Text'])

net = LSTMnet.LSTMnet(vocab = len(word2idx), lr = .1)
length = trainSet.shape[0]
count = 1

test = True

if test == True:
    for i in trainSet:
        print('Importing raw sentence: ', i)
        sentence = i.split()
        if len(sentence) <= 1:
            print('Invalid length, skipping item')
            continue
        print('Splitting sentence')
        input, labels = sentence[:-1], sentence[1:]
        input, labels = dataUtils.prepareSequence(input, word2idx), dataUtils.prepareSequence(labels, word2idx)
        print('Training item')
        net.train(input, labels)
        print('Moving to next item, {}/{}\n'.format(count, length))
        count += 1
