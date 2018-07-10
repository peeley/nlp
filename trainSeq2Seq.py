import langModel, seq2seq, torch, random, datetime, time
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt

maxWords = 10
minWords = 0

corpus = pd.read_csv('data/spa-eng/spa.txt', sep = '\t', lineterminator = '\n', names = ['eng','spa'])
trainingData = corpus.iloc[:10000]

eng = langModel.langModel('english')
spa = langModel.langModel('spanish')

#eng.addEmbedding('data/embeddings/english/', 'glove.6B.300d.txt')
for row in range(trainingData.shape[0]):
    eng.addSentence(langModel.normalize(trainingData.iloc[row]['eng']))
    spa.addSentence(langModel.normalize(trainingData.iloc[row]['spa']))

train = True

if train == True:
    epochs = 15
    recordIndex = 0
    recordInterval = 50
    teacherForceRatio = .5
    loss_fn = nn.NLLLoss()

    encoder = seq2seq.encoder(eng.nWords+1, hiddenSize = 300, lr = .01)
    decoder = seq2seq.decoder(spa.nWords+1, hiddenSize = 300, lr = .01, dropoutProb = .1)
    parameters = filter(lambda p: p.requires_grad, encoder.parameters())
    encoderOptim = torch.optim.SGD(parameters, encoder.lr)
    decoderOptim = torch.optim.SGD(decoder.parameters(), decoder.lr)
    encoderScheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(encoderOptim)
    decoderScheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(decoderOptim)

    losses = []

    startTime = datetime.datetime.now()
    for epoch in range(epochs):
        for row in range(trainingData.shape[0]):
            print('Item #{}/{} \t Epoch {}/{}'.format(row+1, trainingData.shape[0], epoch+1, epochs))
            loss = 0
            inputString = langModel.expandContractions(langModel.normalize(trainingData.iloc[row]['eng']))
            targetString = langModel.normalize(trainingData.iloc[row]['spa'])
            inputTensor, targetTensor = langModel.tensorFromPair(eng, spa, inputString, targetString, True)
            encoderOptim.zero_grad()
            decoderOptim.zero_grad()

            encoderHidden = encoder.initHidden()
            encoderOutputs = torch.zeros(inputTensor.shape[0], encoder.hiddenSize)
            print('Encoding sentence: \t', inputString)
            for inputLetter in range(inputTensor.shape[0]):
                encoderOutput, encoderHidden = encoder(inputTensor[inputLetter], encoderHidden)
                encoderOutputs[inputLetter] = encoderOutput[0,0]
            
            decoderInput = torch.tensor([[spa.SOS]])
            decoderHidden = encoderHidden

            teacherForce = True if random.random() < teacherForceRatio else False
            
            print('Target sentence: \t', targetString)
            decodedString = []
            if teacherForce:
                for targetLetter in range(targetTensor.shape[0]):
                    decoderOutput, decoderHidden = decoder(decoderInput, decoderHidden)
                    topv, topi = decoderOutput.topk(1)
                    loss += loss_fn(decoderOutput, targetTensor[targetLetter])
                    decoderInput = topi.squeeze().detach()
                    if decoderInput.item() == spa.EOS:
                        decodedString.append('/end/')
                        break
                    decodedString.append(spa.idx2word[decoderInput.item()])
            else:
                for targetLetter in range(targetTensor.shape[0]):
                    decoderOutput, decoderHidden = decoder(decoderInput, decoderHidden)
                    loss += loss_fn(decoderOutput, targetTensor[targetLetter])
                    decoderInput = targetTensor[targetLetter]
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
        encoderScheduler.step(loss)
        decoderScheduler.step(loss)

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
    encoder = seq2seq.encoder(eng.nWords+1, hiddenSize = 300, lr = .01)
    decoder = seq2seq.decoder(spa.nWords+1, hiddenSize = 300, lr = .01, dropoutProb = .1)
    encoder.load_state_dict(torch.load('encoder.pt'))
    decoder.load_state_dict(torch.load('decoder.pt'))

def evaluate(rawString, testTarget = None):
    with torch.no_grad():
        for item in range(len(rawString)):
            inputString = langModel.expandContractions(langModel.normalize(rawString[item]))
            print('\nTest sentence: \t', inputString)
            inputSentence = langModel.tensorFromSentence(eng, inputString, False)
            if inputSentence[0] == -1:
                break
            inputLength = len(inputSentence)
            encoderHidden = encoder.initHidden()
            encoderOutputs = torch.zeros(inputLength, encoder.hiddenSize)
            for word in range(inputLength):
                encoderOutput, encoderHidden = encoder(inputSentence[word], encoderHidden)
                encoderOutputs[word] = encoderOutput[0,0]

            decoderInput = torch.tensor([[0]])
            decoderHidden = encoderHidden
            decodedWords = []
            
            for letter in range(inputLength):
                decoderOutput, decoderHidden = decoder(decoderInput, decoderHidden)
                topv, topi = decoderOutput.data.topk(1)
                if topi.item() == 1:
                    #decodedWords.append('/end/')
                    break
                else:
                    decodedWords.append(spa.idx2word[topi.item()])
                decoderInput = topi.squeeze().detach()
            if testTarget:
                print('Target: \t', testTarget)
            print('Translated: \t', ' '.join(decodedWords),'\n')

sample = False

if sample == True:
    while True:
        sampleIndex = int(random.random()*trainingData.shape[0])
        testData = [trainingData.iloc[sampleIndex]['eng']]
        target = trainingData.iloc[sampleIndex]['spa']
        evaluate(testData, target)
        time.sleep(1)

else:
    while True:
        testString = input('Enter text to be translated: ')
        testData = [testString]
        evaluate(testData)

