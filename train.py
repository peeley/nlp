import recurrentNet, dataUtils
import matplotlib.pyplot as plt

net = recurrentNet.RNN()
data = dataUtils.constructJoke()
trainSet = data[:5000]
testSet = data[trainSet.shape[0]:300]
losses = []

for i in range(net.epochs):
    for batch in trainSet:
        input = dataUtils.inputTensor(batch)
        target = dataUtils.targetTensor(batch)
        print('Input: ' , batch)
        output, loss = net.train(input, target)
        print('Output: ', output)
        print('Loss: ', loss.item(), '\n')
        losses.append(loss)

plt.plot(losses)
plt.show()
