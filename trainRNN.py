import recurrentNet, dataUtils, random, torch
import matplotlib.pyplot as plt

net = recurrentNet.RNN()

data = dataUtils.constructPokemon()
trainSet = data[:]
losses = []

train = True
count = 0
if train:
    for i in range(net.epochs):
        for batch in trainSet:
            input = dataUtils.inputTensor(batch)
            target = dataUtils.targetTensor(batch)
            print('\tItem #{}/{} \t Epoch #{}'.format(count, trainSet.shape[0], i+1))
            print('Input:  ' , batch)
            output, loss = net.train(input)
            print('Output: ', output)
            print('Loss: ', loss.item(), '\n')
            losses.append(loss)
            count += 1
        count = 1

def sampleNet(sampleSize, maxLength):
    print("SAMPLING NETWORK \n")
    for i in range(sampleSize):
        noise = random.randint(0, dataUtils.n_letters)
        inputLetter = dataUtils.all_letters[noise]
        input = dataUtils.letterToTensor(inputLetter).view(1, -1)
        outputString = inputLetter
        outputChar = ''
        lastHidden = torch.rand(net.hidden_size)
        for j in range(maxLength):
            if outputChar == '>':
                break
            lastHidden, output = net(input, lastHidden)
            outputChar = dataUtils.all_letters[torch.argmax(output)]
            outputString += outputChar
            input = output.view(1,-1)
        print("Sample #{}: \t{}".format(i+1, outputString))

sampleNet(sampleSize=10, maxLength= 10)

plt.plot(losses)
plt.show()
