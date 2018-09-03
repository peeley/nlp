import langModel, seq2seq, torch, random, datetime, dataUtils, evaluateSeq2Seq, json
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt

size = -1
epochs = 30
dataSentenceLength = 20
maxWords = 100
hSize = 128
layers = 2
batch = 1
reverse = False

engBible = 'data/inupiaq/bible_eng_bpe'
ipqBible = 'data/inupiaq/bible_ipq_bpe'
engBibleVal = 'data/inupiaq/bible_eng_val_bpe'
ipqBibleVal = 'data/inupiaq/bible_ipq_val_bpe'

if reverse:
    trainingData, testLang, targetLang = dataUtils.loadTrainingData(size, dataSentenceLength, ipqBible, engBible, 'ipq', 'eng')
else:
    trainingData, testLang, targetLang = dataUtils.loadTrainingData(size, dataSentenceLength, engBible, ipqBible, 'eng', 'ipq')
print('Translating from {} to {}'.format(testLang.name, targetLang.name))

testData = dataUtils.loadTestData(engBibleVal, ipqBibleVal, testLang, targetLang) 
dataLoader = torch.utils.data.DataLoader(trainingData, shuffle = True, num_workers = 0, batch_size = batch)

train = True
cuda = False
if train == True:
    recordInterval = 25
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
            if teacherForce: # teacher forcing, letters of target sentence are next input of decoder
                for targetLetter in range(targetTensor.shape[0]):
                    decoderOutput, decoderHidden = decoder(decoderInput, decoderHidden, encoderOutputs)
                    loss += loss_fn(decoderOutput, targetTensor[targetLetter])
                    decoderInput = targetTensor[targetLetter]
                    if decoderInput.item() == targetLang.EOS:
                        decodedString.append('/end/')
                        break
                    decodedString.append(targetLang.idx2word[decoderInput.item()])
            else:  # no teacher forcing, outputs are fed as inputs of decoder 
                for targetLetter in range(targetTensor.shape[0]):
                    decoderOutput, decoderHidden = decoder(decoderInput, decoderHidden, encoderOutputs)
                    topv, topi = decoderOutput.topk(1)
                    loss += loss_fn(decoderOutput, targetTensor[targetLetter])
                    decoderInput = topi.squeeze().detach()
                    if decoderInput.item() == targetLang.EOS:
                        decodedString.append('/end/')
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
    torch.save(encoder, 'encoder.pt')
    torch.save(decoder, 'decoder.pt')
    settingsDict = {
            'maxWords' : maxWords,
            'hSize' : hSize,
            'layers' : layers,
            }
    with open('params.json', 'w+') as params:
        json.dump(settingsDict, params)
    print('Models saved to disk.')
    evaluateSeq2Seq.testBLEU(testData, encoder, decoder, testLang, targetLang)
    print('Final loss: \t', losses[-1].item())
    print('Elapsed time: \t', elapsedTime)
    print('Writing models to disk...')
    plt.plot(losses, label = "Losses")
    plt.show()
    plt.savefig('results.png')


