import recurrentNet, dataUtils, random, torch
import matplotlib.pyplot as plt

net = recurrentNet.RNN()

data = dataUtils.constructPokemon()
trainSet = data[:50]
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
        inputLetter = dataUtils.all_letters[random.randint(0, dataUtils.n_letters-1)]
        input = dataUtils.letterToTensor(inputLetter).view(1, -1)
        outputString = inputLetter
        lastHidden = torch.rand(net.hidden_size)
        print(lastHidden.shape)
        for j in range(maxLength):
            lastHidden, output = net(input, lastHidden)
            outputChar = dataUtils.all_letters[torch.argmax(output)]
            outputString += outputChar
            input = output
        print("Sample #{}: {}\n".format(i+1, outputString))

sampleNet(10, 10)
