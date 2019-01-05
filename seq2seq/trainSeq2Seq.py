#!/usr/bin/python

from src import langModel, seq2seq, dataUtils
import torch, random, datetime, json, pickle, argparse
import torch.nn as nn

parser = argparse.ArgumentParser(description = "Script to train Qalgu translator.")
parser.add_argument('--size', type=int, default = 10000, help="Default: 10000")
parser.add_argument('--epochs', type=int, default = 15, help="Default: 15")
parser.add_argument('--maxWords', type=int, default = 25, help="Default: 50")
parser.add_argument('--hSize', type=int, default = 256, help="Default: 1024")
parser.add_argument('--layers', type=int, default = 4, help="Default: 4")
parser.add_argument('--batch', type=int, default = 64, help="Default: 1")
parser.add_argument('--reverse', type=bool, default = False, help="Default: False")
parser.add_argument('--lr', type=float, default=.001, help="Default: .001")
args = parser.parse_args()

settingsDict = {}
for arg in vars(args):
    value = getattr(args,arg)
    settingsDict[arg] = value
    print(f'Writing parameter {arg} = {value}')
with open('src/models/params.json', 'w+') as params:
    print('\nWriting parameters to disk...')
    json.dump(settingsDict, params)
    print('Saved parameters to disk.')

from src import evaluateUtils

testDataFile = 'data/inupiaq/data_eng_train'
targetDataFile = 'data/inupiaq/data_ipq_train_bpe'
testDataValFile = 'data/inupiaq/data_eng_val'
targetDataValFile = 'data/inupiaq/data_ipq_val_bpe' 

testLang = langModel.langModel('eng')
targetLang = langModel.langModel('ipq')
if args.reverse:
    testLang, targetLang = targetLang, testLang
    testData, targetData = targetData, testData
    testDataVal, targetDataVal = targetDataVal, testDataVal

trainingData = dataUtils.loadData(args.size, args.maxWords, testDataFile, targetDataFile, testLang, targetLang, train=True)
testData = dataUtils.loadData(100, args.maxWords, testDataFile, targetDataFile, testLang, targetLang, train=False)
dataLoader = torch.utils.data.DataLoader(trainingData, shuffle = True, num_workers = 4, 
                                         batch_size = args.batch, drop_last = True)
testLoader = torch.utils.data.DataLoader(testData, shuffle = True, num_workers = 4, batch_size = 1)

cuda = False
if torch.cuda.is_available():
    device = torch.device('cuda')
    cuda = True
else:
    device = torch.device('cpu')
print(f'Processing with device {device}.\n')

encoder = seq2seq.encoder(testLang.nWords, hiddenSize=args.hSize, numLayers = args.layers).to(device)
decoder = seq2seq.bahdanauDecoder(targetLang.nWords, hiddenSize=args.hSize, 
                              maxLength=args.maxWords, numLayers = args.layers).to(device)

loss_fn = nn.CrossEntropyLoss(ignore_index = testLang.PAD, reduction = 'sum')
encoderOptim = torch.optim.Adam(encoder.parameters(), lr= args.lr)
decoderOptim = torch.optim.Adam(decoder.parameters(), lr= args.lr)

teacherForceRatio = .5

startTime = datetime.datetime.now()
for epoch in range(args.epochs):
    epochTime = datetime.datetime.now()
    epochLoss = 0
    for row, item in enumerate(dataLoader):
        stepStartTime = datetime.datetime.now()
        inputTensor, targetTensor = item[0].to(device), item[1].to(device)
        inputTensor, targetTensor = inputTensor.transpose(0,1), targetTensor.transpose(0,1)
        seqLengths = inputTensor.shape[0]
        batchSize = inputTensor.shape[1]
        inputLine, targetLine = item[2][0], item[3][0]
        loss = 0
        encoderOptim.zero_grad()
        decoderOptim.zero_grad()
        
        encoderOutput, encoderHidden = encoder(inputTensor, None)
        decoderInput = torch.LongTensor([testLang.SOS] * batchSize).to(device)
        decoderHidden = encoderHidden[:args.layers]

        teacherForce = random.random() < teacherForceRatio

        for currentWord in range(seqLengths):
            decoderOutput, decoderHidden = decoder(decoderInput, decoderHidden, encoderOutput)
            thisLoss = loss_fn(decoderOutput[:].squeeze(1), targetTensor[currentWord, :].squeeze(1))
            loss += thisLoss
            if teacherForce:
                decoderInput = targetTensor[currentWord]
            else:
                topv, topi = decoderOutput.topk(1)
                decoderInput = topi.squeeze().detach().view(batchSize)

        loss.backward()
        nn.utils.clip_grad_norm_(decoder.parameters(), 50)
        nn.utils.clip_grad_norm_(encoder.parameters(), 50)
        encoderOptim.step()
        decoderOptim.step()
        epochLoss += loss.item()

        stepTime = datetime.datetime.now() - stepStartTime
    epochLoss = epochLoss / args.size
    epochTime = datetime.datetime.now() - epochTime
    bleu = evaluateUtils.testBLEU(testLoader, encoder, decoder, testLang, targetLang, False)

    print(f"Epoch: {epoch+1}\tLoss: {epochLoss:.5f}\tEpoch Time: {epochTime}\tStep Time: {stepTime}\tBLEU: {bleu*100:.5f}")
endTime = datetime.datetime.now()
elapsedTime = endTime - startTime
print('Writing language models to disk...')
with open(f"src/models/{testLang.name}.p", 'wb') as engFile:
    pickle.dump(testLang, engFile)
with open(f"src/models/{targetLang.name}.p", 'wb') as ipqFile:
    pickle.dump(targetLang, ipqFile)
print('Language models saved to disk.')
print('Writing models to disk...')
torch.save(encoder, 'src/models/encoder.pt')
torch.save(decoder, 'src/models/decoder.pt')
print('Models saved to disk.\n')

evaluateUtils.testBLEU(testLoader, encoder, decoder, testLang, targetLang, True)
print('Final loss: \t', epochLoss)
print('Elapsed time: \t', elapsedTime)
