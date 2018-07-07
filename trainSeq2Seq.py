import langModel, seq2seq, torch, random, datetime, time
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt

maxWords = 10
minWords = 0

corpus = pd.read_csv('data/spa-eng/spa.txt', sep = '\t', lineterminator = '\n', names = ['eng','spa'])

mask = ((corpus['eng'].str.split().apply(len) < maxWords) & (corpus['eng'].str.split().apply(len) > minWords) ) & ((corpus['spa'].str.split().apply(len) < maxWords) & (corpus['spa'].str.split().apply(len) > minWords))
corpus = corpus[mask]
trainingData = corpus.iloc[:500]

eng = langModel.langModel('english')
spa = langModel.langModel('spanish')
for row in range(trainingData.shape[0]):
    eng.addSentence(langModel.normalize(trainingData.iloc[row]['eng']))
    spa.addSentence(langModel.normalize(trainingData.iloc[row]['spa']))

glove = False
train = True

if train == True:
    epochs = 10
    recordIndex = 0
    recordInterval = 50
    teacherForceRatio = .5
    loss_fn = nn.NLLLoss()

    encoder = seq2seq.encoder(eng.nWords, hiddenSize = 300, lr = .01)
    decoder = seq2seq.decoder(spa.nWords, hiddenSize = 300, lr = .01, dropoutProb = .1)
    if glove:
        parameters = filter(lambda p: p.requires_grad, encoder.parameters())
        encoderOptim = torch.optim.SGD(parameters, encoder.lr, glove)
    else:
        encoderOptim = torch.optim.SGD(encoder.parameters(), encoder.lr, glove)
    decoderOptim = torch.optim.SGD(decoder.parameters(), decoder.lr)

    losses = []

    startTime = datetime.datetime.now()
    for epoch in range(epochs):
        for row in range(trainingData.shape[0]):
            print('Item #{}/{} \t Epoch {}/{}'.format(row+1, trainingData.shape[0], epoch+1, epochs))
            loss = 0
            inputString = langModel.normalize(trainingData.iloc[row]['eng'])
            targetString = langModel.normalize(trainingData.iloc[row]['spa'])
            input, target = langModel.tensorFromPair(eng, spa, inputString, targetString)
            encoderOptim.zero_grad()
            decoderOptim.zero_grad()

            encoderHidden = encoder.initHidden()
            encoderOutputs = torch.zeros(maxWords, encoder.hiddenSize)
            print('Encoding sentence: \t', inputString)
            for inputLetter in range(input.shape[0]):
                encoderOutput, encoderHidden = encoder(input[inputLetter], encoderHidden)
                encoderOutputs[inputLetter] = encoderOutput[0,0]
            
            decoderInput = torch.tensor([[0]])
            decoderHidden = encoderHidden

            teacherForce = True if random.random() < teacherForceRatio else False
            
            print('Target sentence: \t', targetString)
            decodedString = []
            if teacherForce:
                for targetLetter in range(target.shape[0]):
                    decoderOutput, decoderHidden = decoder(decoderInput, decoderHidden)
                    topv, topi = decoderOutput.topk(1)
                    loss += loss_fn(decoderOutput, target[targetLetter])
                    decoderInput = topi.squeeze().detach()
                    if decoderInput.item() == 1:
                        decodedString.append('<EOS>')
                        break
                    decodedString.append(spa.idx2word[decoderInput.item()])
            else:
                for targetLetter in range(target.shape[0]):
                    decoderOutput, decoderHidden = decoder(decoderInput, decoderHidden)
                    loss += loss_fn(decoderOutput, target[targetLetter])
                    decoderInput = target[targetLetter]
                    if decoderInput.item() == 1:
                        break
                    decodedString.append(spa.idx2word[decoderInput.item()])
            print('Translated sentence: \t', ' '.join(decodedString))

            loss.backward()
            encoderOptim.step()
            decoderOptim.step()
            recordIndex += 1
            if recordIndex % recordInterval == 0:
                losses.append(loss)
            print('Loss: ', loss.item(), '\n')
    endTime = datetime.datetime.now()
    elapsedTime = endTime - startTime
    print('Elapsed time: ', elapsedTime)
    plt.plot(losses)
    plt.show()
    print('Writing models to disk...')
    torch.save(encoder.state_dict(), 'encoder.pt')
    torch.save(decoder.state_dict(), 'decoder.pt')
    print('Models saved to disk.')

else:
    encoder = seq2seq.encoder(eng.nWords, hiddenSize = 256, lr = .01)
    decoder = seq2seq.decoder(spa.nWords, hiddenSize = 256, lr = .01, dropoutProb = .1)
    encoder.load_state_dict(torch.load('encoder.pt'))
    decoder.load_state_dict(torch.load('decoder.pt'))

def evaluate(input):
    with torch.no_grad():
        for item in range(len(input)):
            inputString = langModel.normalize(input[item])
            print('\nTest sentence: \t', inputString)
            try:
                input = langModel.tensorFromSentence(eng, inputString)
            except KeyError as e:
                print('\nERROR: Word not in vocabulary, ', str(e), '\n')
                break
            inputLength = input.shape[0]
            encoderHidden = encoder.initHidden()
            encoderOutputs = torch.zeros(maxWords, encoder.hiddenSize)
            for letter in range(inputLength):
                encoderOutput, encoderHidden = encoder(input[letter], encoderHidden)
                encoderOutputs[letter] = encoderOutput[0,0]

            decoderInput = torch.tensor([[0]])
            decoderHidden = encoderHidden
            decodedWords = []
            
            for letter in range(maxWords):
                decoderOutput, decoderHidden = decoder(decoderInput, decoderHidden)
                topv, topi = decoderOutput.data.topk(1)
                if topi.item() == 1:
                    #decodedWords.append('<EOS>')
                    break
                else:
                    decodedWords.append(spa.idx2word[topi.item()])
                decoderInput = topi.squeeze().detach()
            if sample == True:
                print('Target: \t', testTarget)
            print('Translated: \t', ' '.join(decodedWords),'\n')

sample = False

if sample == True:
    while True:
        sampleIndex = int(random.random()*trainingData.shape[0])
        testData = [trainingData.iloc[sampleIndex]['eng']]
        testTarget = trainingData.iloc[sampleIndex]['spa']
        evaluate(testData)
        time.sleep(1)

else:
    while True:
        testData = [langModel.normalize(input('Enter text to be translated: '))]
        evaluate(testData)

