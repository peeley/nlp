#!/usr/bin/python

import langModel, seq2seq, torch, random, datetime, dataUtils, evaluateSeq2Seq, json, pickle, argparse
import torch.nn as nn

parser = argparse.ArgumentParser(description = "Script to train Qalgu translator.")
parser.add_argument('--size', type=int, default = -1, help="Default: -1")
parser.add_argument('--epochs', type=int, default = 15, help="Default: 15")
parser.add_argument('--dataSentenceLength', type=int, default = 15, help="Default: 15")
parser.add_argument('--maxWords', type=int, default = 50, help="Default: 50")
parser.add_argument('--hSize', type=int, default = 1024, help="Default: 1024")
parser.add_argument('--layers', type=int, default = 4, help="Default: 4")
parser.add_argument('--batch', type=int, default = 1, help="Default: 1")
parser.add_argument('--reverse', type=bool, default = False, help="Default: False")
parser.add_argument('--lengthScheduling', type=int, default = False, help="Default: False")
args = parser.parse_args()
print('Running with following settings: \n', args)

# Change length of sentences based on how long we've been training
lengthSchedule = {0: args.dataSentenceLength,
                  (args.epochs//3): args.dataSentenceLength + 10, 
                  (args.epochs//2): args.dataSentenceLength + 20, 
                  (args.epochs//1.5): args.dataSentenceLength + 30} 
lengthScheduling = False

testData = 'data/de-en/train.tok.clean.bpe.32000.en'
targetData = 'data/de-en/train.tok.clean.bpe.32000.de'
testDataVal = 'data/de-en/newstest2011.tok.bpe.32000.en'
targetDataVal = 'data/de-en/newstest2011.tok.bpe.32000.de'

testLang = langModel.langModel('eng')
targetLang = langModel.langModel('ipq')
if args.reverse:
    testLang, targetLang = targetLang, testLang
    testData, targetData = targetData, testData
    testDataVal, targetDataVal = targetDataVal, testDataVal
trainingData = dataUtils.loadTrainingData(args.size, args.dataSentenceLength, testData, targetData, testLang, targetLang)
testData = dataUtils.loadTestData(testDataVal, targetDataVal, testLang, targetLang) 
dataLoader = torch.utils.data.DataLoader(trainingData, shuffle = True, num_workers = 0, batch_size = args.batch, drop_last = True)

cuda = False
teacherForceRatio = .33
loss_fn = nn.NLLLoss()
if torch.cuda.is_available():
    device = torch.device('cuda')
    cuda = True
    print('PROCESSING WITH CUDA DEVICE ', device)
else:
    device = torch.device('cpu')
    print('PROCESSING WITH CPU\n')

encoder = seq2seq.encoder(testLang.nWords+1, hiddenSize=args.hSize, 
                          lr = .1, numLayers = args.layers, 
                          batchSize = args.batch).to(device)
decoder = seq2seq.attnDecoder(targetLang.nWords+1, hiddenSize=args.hSize, 
                              lr = .1, dropoutProb = .001, 
                              maxLength=args.maxWords, numLayers = args.layers,
                              batchSize = args.batch).to(device)
encoderOptim = torch.optim.SGD(encoder.parameters(), encoder.lr, momentum= .9, nesterov= True)
decoderOptim = torch.optim.SGD(decoder.parameters(), decoder.lr, momentum= .9, nesterov= True)
#encoderScheduler = torch.optim.lr_scheduler.ExponentialLR(encoderOptim, gamma = .9)
#decoderScheduler = torch.optim.lr_scheduler.ExponentialLR(decoderOptim, gamma = .9)
startTime = datetime.datetime.now()
for epoch in range(args.epochs):
    epochLoss = 0
    if lengthScheduling:
        if epoch in lengthSchedule.keys():
            print("Increasing sentence length")
            args.dataSentenceLength = lengthSchedule[epoch]
            trainingData = dataUtils.loadTrainingData(args.size, args.dataSentenceLength, 
                                                      testData, targetData, testLang, targetLang)
            dataLoader = torch.utils.data.DataLoader(trainingData, shuffle = True, 
                                                     num_workers = numWorkers, _size = args.batch)
            print("Created new dataset")
    for row, item in enumerate(dataLoader):
        stepStartTime = datetime.datetime.now()
        inputTensor, targetTensor = item[0].to(device), item[1].to(device)
        inputLine, targetLine = item[2][0], item[3][0]
        loss = 0
        
        encoderOptim.zero_grad()
        decoderOptim.zero_grad()

        encoderHidden = seq2seq.initHidden(cuda, args.hSize, args.layers*2, args.batch)
        encoderOutputs = torch.zeros(args.batch, args.maxWords, args.hSize * 2).to(device)
        for inputLetter in range(inputTensor.shape[1]):
            encoderOutput, encoderHidden = encoder(inputTensor[:,inputLetter], encoderHidden)
            encoderOutputs[:, inputLetter] = encoderOutput[:, 0]
        
        decoderInput = torch.zeros([args.batch, 1], dtype = torch.long).to(device)
        decoderHidden = encoderHidden

        teacherForce = True if random.random() < teacherForceRatio else False

        decodedString = []
        for targetLetter in range(targetTensor.shape[1]):
            decoderOutput, decoderHidden = decoder(decoderInput, decoderHidden, encoderOutputs)
            decoderOutput = decoderOutput.view(args.batch, targetLang.nWords+1)
            thisLoss = loss_fn(decoderOutput[:], targetTensor[:, targetLetter].view(args.batch,))
            loss += thisLoss
            if teacherForce:
                decoderInput = targetTensor[:, targetLetter]
            else:
                topv, topi = decoderOutput.topk(1)
                decoderInput = topi.squeeze().detach()
            decodedString.append(targetLang.idx2word[decoderInput[0].item()])

        loss.backward()
        nn.utils.clip_grad_norm_(decoder.parameters(), 10)
        encoderOptim.step()
        decoderOptim.step()
        epochLoss += loss

        stepEndTime = datetime.datetime.now()
        stepTime = stepEndTime - stepStartTime
    print('Epoch: {} \t Loss: {}'.format(epoch+1, epochLoss))
    #encoderScheduler.step()
    #decoderScheduler.step()
endTime = datetime.datetime.now()
elapsedTime = endTime - startTime
settingsDict = {}

for arg in vars(args):
    settingsDict[arg] = getattr(args, arg)

with open('params.json', 'w+') as params:
    print('\nWriting parameters to disk...')
    json.dump(settingsDict, params)
    print('Saved parameters to disk.')

print('Writing language models to disk...')
with open(testLang.name+'.p', 'wb') as engFile:
    pickle.dump(testLang, engFile)
with open(targetLang.name+'.p', 'wb') as ipqFile:
    pickle.dump(targetLang, ipqFile)
print('Language models saved to disk.')
print('Writing models to disk...')
torch.save(encoder, 'encoder.pt')
torch.save(decoder, 'decoder.pt')
print('Models saved to disk.\n')
evaluateSeq2Seq.testBLEU(testData, encoder, decoder, testLang, targetLang)
print('Final loss: \t', loss.item())
print('Elapsed time: \t', elapsedTime)
