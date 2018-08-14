import langModel, seq2seq, torch, random, datetime, dataUtils, evaluateSeq2Seq
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt

maxWords = 25
size = 200
corpus, eng, de = dataUtils.loadEnDe(size+100, 15)
trainingData = corpus.iloc[:size]
testData = corpus.iloc[size:size+100]
train = True
cuda = False
hiddenSizes = {'debug':300, 'prod':1024}
if train == True:
    epochs = 20
    recordIndex = 0
    recordInterval = 25
    teacherForceRatio = .5
    loss_fn = nn.NLLLoss()
    bleuAVG = 0
    bleuScores = []

    encoder = seq2seq.encoder(eng.nWords, 256, lr = .001, numLayers = 2)
    decoder = seq2seq.attnDecoder(de.nWords, 256, lr = .001, dropoutProb = .001, maxLength=maxWords, numLayers = encoder.numLayers * 2)
    #encoderOptim = torch.optim.SGD(encoder.parameters(), encoder.lr, momentum = .9)
    #decoderOptim = torch.optim.SGD(decoder.parameters(), decoder.lr, momentum = .9)
    encoderOptim = torch.optim.Adam(encoder.parameters(), encoder.lr)
    decoderOptim = torch.optim.Adam(decoder.parameters(), encoder.lr)
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
        for row in range(size):
            sample = int(size * random.random())
            print('Item #{}/{} \t Sample Row: {} \t Epoch {}/{}'.format(row+1, trainingData.shape[0], sample, epoch+1, epochs))
            loss = 0
            inputString = langModel.expandContractions(langModel.normalize(trainingData.iloc[sample]['eng']))
            targetString = langModel.normalize(trainingData.iloc[sample]['de'])
            inputTensor, targetTensor = langModel.tensorFromPair(eng, de, inputString, targetString, train)
            if cuda:
                inputTensor, targetTensor = inputTensor.to(device), targetTensor.to(device)
            
            encoderOptim.zero_grad()
            decoderOptim.zero_grad()

            encoderHidden = seq2seq.initHidden(cuda, encoder.hiddenSize, decoder.numLayers)
            # encoderOutputs dimension 1 is hiddenSize * 2 for bidiredtionality
            encoderOutputs = torch.zeros(decoder.maxLength, encoder.hiddenSize * 2)
            if cuda:
                encoderOutputs = encoderOutputs.to(device)

            print('Encoding sentence: \t', inputString.encode('utf8'))
            for inputLetter in range(inputTensor.shape[0]):
                encoderOutput, encoderHidden = encoder(inputTensor[inputLetter], encoderHidden)
                encoderOutputs[inputLetter] = encoderOutput[0,0]
            
            decoderInput = torch.tensor([[de.SOS]])
            if cuda:
                decoderInput = decoderInput.to(device)
            decoderHidden = encoderHidden

            teacherForce = True if random.random() < teacherForceRatio else False

            print('Target sentence: \t', targetString.encode('utf8'))
            decodedString = []
            if teacherForce: # teacher forcing, letters of target sentence are next input of decoder
                for targetLetter in range(targetTensor.shape[0]):
                    decoderOutput, decoderHidden = decoder(decoderInput, decoderHidden, encoderOutputs)
                    loss += loss_fn(decoderOutput, targetTensor[targetLetter])
                    decoderInput = targetTensor[targetLetter]
                    if decoderInput.item() == 1:
                        break
                    decodedString.append(de.idx2word[decoderInput.item()])
            else:  # no teacher forcing, outputs are fed as inputs of decoder 
                for targetLetter in range(targetTensor.shape[0]):
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
            print('Translated sentence: \t', ' '.join(decodedString).encode('utf8'))

            loss.backward()
            nn.utils.clip_grad_norm(decoder.parameters(), 5)
            encoderOptim.step()
            decoderOptim.step()

            recordIndex += 1
            if recordIndex % recordInterval == 0:
                losses.append(loss)
            print('Loss: \t\t', loss.item(), '\n')
        #encoderScheduler.step(loss)
        #decoderScheduler.step(loss)
    evaluateSeq2Seq.testBLEU(testData, encoder, decoder, eng, de)
    print('Final loss: \t', losses[-1].item())
    endTime = datetime.datetime.now()
    elapsedTime = endTime - startTime
    print('Elapsed time: \t', elapsedTime)
    plt.plot(losses, label = "Losses")
    plt.show()
    plt.savefig('results.png')
    print('Writing models to disk...')
    torch.save(encoder, 'encoder.pt')
    torch.save(decoder, 'decoder.pt')
    print('Models saved to disk.')


