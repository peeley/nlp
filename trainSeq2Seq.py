import langModel, seq2seq, torch, random, datetime, dataUtils, evaluateSeq2Seq
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt

maxWords = 15
size = 100
vocabSize = 10
hSize = 256
layers = 3
batch = 1

trainingData, eng, de = dataUtils.loadEnDe(size, vocabSize)
dataLoader = torch.utils.data.DataLoader(trainingData, shuffle = True, num_workers = 8, batch_size = batch)
testData = dataUtils.loadTestEnDe(vocabSize)[:100]

train = True
cuda = False
hiddenSizes = {'debug':300, 'prod':1024}
if train == True:
    epochs = 40
    recordInterval = 25
    teacherForceRatio = .5
    loss_fn = nn.NLLLoss()
    bleuAVG = 0
    bleuScores = []
    if torch.cuda.is_available():
        device = torch.device('cuda')
        cuda = True
        print('PROCESSING WITH CUDA DEVICE ', device)
    else:
        device = torch.device('cpu')
        print('PROCESSING WITH CPU')

    encoder = seq2seq.encoder(eng.nWords+1, hiddenSize=hSize, lr = .01, numLayers = layers, batchSize = batch).to(device)
    decoder = seq2seq.attnDecoder(de.nWords+1, hiddenSize=hSize, lr = .01, dropoutProb = .001, maxLength=maxWords, numLayers = layers).to(device)
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
            inputTensor, targetTensor = item[0].view( -1, 1).to(device), item[1].view( -1, 1).to(device)
            loss = 0
            print('Item #{}/{} \t Epoch {}/{}'.format(row+1, len(trainingData), epoch+1, epochs))
            
            encoderOptim.zero_grad()
            decoderOptim.zero_grad()

            encoderHidden = seq2seq.initHidden(cuda, hSize, layers*2)
            encoderOutputs = torch.zeros(maxWords, hSize * 2).to(device)

            for inputLetter in range(inputTensor.shape[0]):
                encoderOutput, encoderHidden = encoder(inputTensor[inputLetter], encoderHidden)
                encoderOutputs[inputLetter] = encoderOutput[0,0]
            
            decoderInput = torch.tensor([[de.SOS]]).to(device)
            decoderHidden = encoderHidden

            teacherForce = True if random.random() < teacherForceRatio else False

            decodedString = []
            if teacherForce: # teacher forcing, letters of target sentence are next input of decoder
                for targetLetter in range(targetTensor.shape[0]):
                    decoderOutput, decoderHidden = decoder(decoderInput, decoderHidden, encoderOutputs)
                    loss += loss_fn(decoderOutput, targetTensor[targetLetter])
                    decoderInput = targetTensor[targetLetter]
                    if decoderInput.item() == de.EOS:
                        decodedString.append('/end/')
                        break
                    decodedString.append(de.idx2word[decoderInput.item()])
            else:  # no teacher forcing, outputs are fed as inputs of decoder 
                for targetLetter in range(targetTensor.shape[0]):
                    decoderOutput, decoderHidden = decoder(decoderInput, decoderHidden, encoderOutputs)
                    topv, topi = decoderOutput.topk(1)
                    loss += loss_fn(decoderOutput, targetTensor[targetLetter])
                    decoderInput = topi.squeeze().detach()
                    if decoderInput.item() == de.EOS:
                        decodedString.append('/end/')
                        break
                    decodedString.append(de.idx2word[decoderInput.item()])
            print('Translated sentence: \t', ' '.join(decodedString))

            loss.backward()
            nn.utils.clip_grad_norm_(decoder.parameters(), 10)
            encoderOptim.step()
            decoderOptim.step()

            if row % recordInterval == 0:
                losses.append(loss)
            print('Loss: \t\t', loss.item(), '\n')
        encoderScheduler.step()
        decoderScheduler.step()
    endTime = datetime.datetime.now()
    elapsedTime = endTime - startTime
    print('Elapsed time: \t', elapsedTime)
    print('Writing models to disk...')
    torch.save(encoder, 'encoder.pt')
    torch.save(decoder, 'decoder.pt')
    settingsDict = {
            'maxWords' : maxWords,
            'size' : size,
            'vocabSize' : vocabSize,
            'hSize' : hSize,
            'layers' : layers,
            'batch' : batch
            }
    print('Models saved to disk.')
    print('Final loss: \t', losses[-1].item())
    plt.plot(losses, label = "Losses")
    plt.show()
    plt.savefig('results.png')
    evaluateSeq2Seq.testBLEU(testData, encoder, decoder, eng, de)


