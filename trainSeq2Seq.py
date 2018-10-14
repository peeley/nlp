#!/usr/bin/python

import langModel, seq2seq, torch, random, datetime, dataUtils, evaluateSeq2Seq, json, pickle
import torch.nn as nn
import matplotlib.pyplot as plt

size = 500                   # dataset size
epochs = 20                 # training epochs
dataSentenceLength = 10     # length of sentences in dataset
maxWords = 50               # max length input of encoder
hSize = 256                 # hidden size of encoder/decoder
layers = 4                  # layers of network for encoder/decoder
batch = 1                   # batch size. TODO: find out how to use batch input.
lengthSchedule = {0: dataSentenceLength,
                  (epochs//3): dataSentenceLength + 10, 
                  (epochs//2): dataSentenceLength + 20, 
                  (epochs//1.5): dataSentenceLength + 30} # Change length of sentences based on how long we've been training
lengthScheduling = False

engData = 'data/de-en/train.tok.clean.bpe.32000.en'
ipqData = 'data/de-en/train.tok.clean.bpe.32000.de'
engDataVal = 'data/de-en/newstest2011.tok.bpe.32000.en'
ipqDataVal = 'data/de-en/newstest2011.tok.bpe.32000.de'

testLang = langModel.langModel('eng')
targetLang = langModel.langModel('ipq')

trainingData = dataUtils.loadTrainingData(size, dataSentenceLength, engData, ipqData, testLang, targetLang)

testData = dataUtils.loadTestData(engDataVal, ipqDataVal, testLang, targetLang) 
dataLoader = torch.utils.data.DataLoader(trainingData, shuffle = True, num_workers = 0, batch_size = batch)

cuda = False
recordInterval = 8
teacherForceRatio = .2
loss_fn = nn.NLLLoss()
bleuScores = []
if torch.cuda.is_available():
    device = torch.device('cuda')
    cuda = True
    print('PROCESSING WITH CUDA DEVICE ', device)
else:
    device = torch.device('cpu')
    print('PROCESSING WITH CPU\n')

encoder = seq2seq.encoder(testLang.nWords+1, hiddenSize=hSize, lr = .05, numLayers = layers, batchSize = batch).to(device)
decoder = seq2seq.attnDecoder(targetLang.nWords+1, hiddenSize=hSize, lr = .05, dropoutProb = .001, maxLength=maxWords, numLayers = layers).to(device)
encoderOptim = torch.optim.SGD(encoder.parameters(), encoder.lr, momentum = .9, nesterov = True)
decoderOptim = torch.optim.SGD(decoder.parameters(), encoder.lr, momentum = .9, nesterov = True)
encoderScheduler = torch.optim.lr_scheduler.ExponentialLR(encoderOptim, gamma = .9)
decoderScheduler = torch.optim.lr_scheduler.ExponentialLR(decoderOptim, gamma = .9)
losses = []
encoder = nn.DataParallel(encoder)
decoder = nn.DataParallel(decoder)

startTime = datetime.datetime.now()
for epoch in range(epochs):
    if lengthScheduling:
        if epoch in lengthSchedule.keys():
            print("Increasing sentence length")
            dataSentenceLength = lengthSchedule[epoch]
            trainingData = dataUtils.loadTrainingData(size, dataSentenceLength, engData, ipqData, testLang, targetLang)
            dataLoader = torch.utils.data.DataLoader(trainingData, shuffle = True, num_workers = 0, batch_size = batch)
            print("Created new dataset")
    for row, item in enumerate(dataLoader):
        stepStartTime = datetime.datetime.now()
        inputTensor, targetTensor = item[0].view( -1, 1).to(device), item[1].view( -1, 1).to(device)
        loss = 0
        
        encoderOptim.zero_grad()
        decoderOptim.zero_grad()

        encoderHidden = seq2seq.initHidden(cuda, hSize, layers*2)
        encoderOutputs = torch.zeros(maxWords, hSize * 2).to(device)

        for inputLetter in range(inputTensor.shape[0]):
            encoderOutput, encoderHidden = encoder(inputTensor[inputLetter], encoderHidden)
            encoderOutputs[inputLetter] = encoderOutput[0,0]
        
        decoderInput = torch.tensor([[targetLang.SOS]]).to(device)
        decoderHidden = encoderHidden

        teacherForce = True if random.random() < teacherForceRatio else False

        decodedString = []
        for targetLetter in range(targetTensor.shape[0]):
            decoderOutput, decoderHidden = decoder(decoderInput, decoderHidden, encoderOutputs)
            thisLoss = loss_fn(decoderOutput, targetTensor[targetLetter])
            loss += thisLoss
            if teacherForce:
                decoderInput = targetTensor[targetLetter]
            else:
                topv, topi = decoderOutput.topk(1)
                decoderInput = topi.squeeze().detach()
            if decoderInput.item() == targetLang.EOS:
                decodedString.append('/end/')
                loss += thisLoss * (targetTensor.shape[0] - targetLetter) # increase loss for each word left out
                break
            decodedString.append(targetLang.idx2word[decoderInput.item()])
        print('Translated sentence: \t', ' '.join(decodedString))

        loss.backward()
        nn.utils.clip_grad_norm_(decoder.parameters(), 10)
        encoderOptim.step()
        decoderOptim.step()

        if row % recordInterval == 0:
            losses.append(loss)
        stepEndTime = datetime.datetime.now()
        stepTime = stepEndTime - stepStartTime
        print('Item #{}/{} \t Epoch {}/{}'.format(row+1, len(trainingData), epoch+1, epochs))
        print('Loss: \t\t', loss.item())
        print('Time: \t\t', stepTime, '\n')
    encoderScheduler.step()
    decoderScheduler.step()
endTime = datetime.datetime.now()
elapsedTime = endTime - startTime
settingsDict = {
        'maxWords' : maxWords,
        'hSize' : hSize,
        'layers' : layers,
        'size' : size,
        'dataSentenceLength' : dataSentenceLength
        }
with open('params.json', 'w+') as params:
    print('\nWriting parameters to disk...')
    json.dump(settingsDict, params)
    print('Saved parameters to disk.')

print('Writing language models to disk...')
with open('eng.p', 'wb') as engFile:
    pickle.dump(testLang, engFile)
with open('ipq.p', 'wb') as ipqFile:
    pickle.dump(targetLang, ipqFile)
print('Language models saved to disk.')
print('Writing models to disk...')
torch.save(encoder, 'encoder.pt')
torch.save(decoder, 'decoder.pt')
print('Models saved to disk.\n')
evaluateSeq2Seq.testBLEU(testData, encoder, decoder, testLang, targetLang)
print('Final loss: \t', losses[-1].item())
print('Elapsed time: \t', elapsedTime)
plt.plot(losses, label = "Losses", color = 'black')
plt.show()
plt.savefig('results.png')
