import langModel, seq2seq, torch, random, datetime, dataUtils
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu

maxWords = 10
size = 100
corpus, eng, de = dataUtils.loadEnDe(size, maxWords)
trainingData = corpus.iloc[:size]

train = True
cuda = False
hiddenSizes = {'debug':300, 'prod':1024}
if train == True:
    epochs = 40
    recordIndex = 0
    recordInterval = 50
    teacherForceRatio = .5
    loss_fn = nn.NLLLoss()
    bleuAVG = 0
    bleuScores = []

    encoder = seq2seq.encoder(eng.nWords+1, hiddenSizes['debug'], lr = .01, numLayers = 2)
    decoder = seq2seq.attnDecoder(de.nWords+1, hiddenSizes['debug'] , lr=.01, dropoutProb=.001, maxLength=maxWords, numLayers = encoder.numLayers * 2)
    parameters = filter(lambda p: p.requires_grad, encoder.parameters())
    encoderOptim = torch.optim.SGD(parameters, encoder.lr, momentum = .9)
    decoderOptim = torch.optim.SGD(decoder.parameters(), decoder.lr, momentum = .9)
    encoderScheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(encoderOptim)
    decoderScheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(decoderOptim)
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        encoder.to(device)
        decoder.to(device)
        cuda = True
        print('PROCESSING WITH CUDA DEVICE ', device)
    losses = []

    startTime = datetime.datetime.now()
    for epoch in range(epochs):
        for row in range(trainingData.shape[0]):
            print('Item #{}/{} \t Epoch {}/{}'.format(row+1, trainingData.shape[0], epoch+1, epochs))
            loss = 0
            inputString = langModel.expandContractions(langModel.normalize(trainingData.iloc[row]['eng']))
            targetString = langModel.normalize(trainingData.iloc[row]['de'])
            inputTensor, targetTensor = langModel.tensorFromPair(eng, de, inputString, targetString, train)
            if cuda:
                inputTensor, targetTensor = inputTensor.to(device), targetTensor.to(device)
            
            encoderOptim.zero_grad()
            decoderOptim.zero_grad()

            encoderHidden = seq2seq.initHidden(cuda, encoder.hiddenSize, decoder.numLayers)
            # encoderOutputs dimension 1 is hiddenSize * 2 for bidiredtionality
            encoderOutputs = torch.zeros(maxWords, encoder.hiddenSize * 2)
            if cuda:
                encoderOutputs = encoderOutputs.to(device)

            print('Encoding sentence: \t', inputString.encode('utf8'))
            for inputLetter in range(inputTensor.shape[0]):
                if cuda:
                    encoderOutput, encoderHidden = encoder(inputTensor[inputLetter], encoderHidden)
                else:
                    encoderOutput, encoderHidden = encoder(inputTensor[inputLetter], encoderHidden)
                encoderOutputs[inputLetter] = encoderOutput[0,0]
            
            decoderInput = torch.tensor([[de.SOS]])
            if cuda:
                decoderInput = decoderInput.to(device)
            decoderHidden = encoderHidden

            teacherForce = True if random.random() < teacherForceRatio else False

            print('Target sentence: \t', targetString.encode('utf8'))
            decodedString = []
            if teacherForce:
                for targetLetter in range(targetTensor.shape[0]):
                    if cuda:
                        decoderOutput, decoderHidden = decoder(decoderInput, decoderHidden, encoderOutputs)
                    else:
                        decoderOutput, decoderHidden = decoder(decoderInput, decoderHidden, encoderOutputs)
                    if cuda:
                        decoderOutput = decoderOutput.to(device)
                    topv, topi = decoderOutput.topk(1)
                    loss += loss_fn(decoderOutput, targetTensor[targetLetter])
                    decoderInput = topi.squeeze().detach()
                    if decoderInput.item() == de.EOS:
                        decodedString.append('/end/')
                        break
                    decodedString.append(de.idx2word[decoderInput.item()])
            else:
                for targetLetter in range(targetTensor.shape[0]):
                    if cuda:
                        decoderOutput, decoderHidden = decoder(decoderInput, decoderHidden, encoderOutputs)
                    else:
                        decoderOutput, decoderHidden = decoder(decoderInput, decoderHidden, encoderOutputs)
                    loss += loss_fn(decoderOutput, targetTensor[targetLetter])
                    decoderInput = targetTensor[targetLetter]
                    if decoderInput.item() == 1:
                        break
                    decodedString.append(de.idx2word[decoderInput.item()])
            print('Translated sentence: \t', ' '.join(decodedString).encode('utf8'))

            loss.backward()
            encoderOptim.step()
            decoderOptim.step()

            recordIndex += 1
            losses.append(loss)
            print('Loss: \t\t', loss.item())
            if '' in decodedString:
                decodedString = list(filter(None, decodedString))
            if len(decodedString) == 0:
                bleu = 0
            else:
                    bleu = sentence_bleu([targetString.split()], decodedString)
            print('BLEU Score: \t', bleu)
            bleuAVG = ((bleuAVG + bleu) / len(losses)) * 100
            bleuScores.append(bleuAVG)
            print('BLEU Average: \t', bleuAVG, '\n')
        encoderScheduler.step(loss)
        decoderScheduler.step(loss)

    endTime = datetime.datetime.now()
    elapsedTime = endTime - startTime
    print('Elapsed time: ', elapsedTime)
    plt.plot(losses, label = "Losses")
    plt.plot(bleuScores, label = "BLEU")
    plt.show()
    plt.savefig('results.png')
    print('Writing models to disk...')
    torch.save(encoder, 'encoder.pt')
    torch.save(decoder, 'decoder.pt')
    print('Models saved to disk.')


