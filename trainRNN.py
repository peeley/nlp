import recurrentNet, dataUtils
import matplotlib.pyplot as plt

net = recurrentNet.RNN()
data = dataUtils.constructJoke()
trainSet = data[:3000]
testSet = data[trainSet.shape[0]:trainSet.shape[0]+300]
losses = []
print(data.shape)

train = True
count = 0
if train:
    for i in range(net.epochs):
        for batch in trainSet:
            input = dataUtils.inputTensor(batch)
            target = dataUtils.targetTensor(batch)
            print('\tItem #{} \t Epoch #{}'.format(count, i))
            print('Input: ' , batch)
            output, loss = net.train(input, target)
            print('Output: ', output)
            print('Loss: ', loss.item(), '\n')
            losses.append(loss)
            count += 1
        count = 0

plt.plot(losses)
plt.show()
