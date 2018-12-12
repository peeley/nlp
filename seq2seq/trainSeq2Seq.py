#!/usr/bin/python

from src import langModel, seq2seq, dataUtils
import torch, random, datetime, json, pickle, argparse, time
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
parser.add_argument('--learningRate', type=float, default=.001, help="Default: .001")
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

from src import evaluateSeq2Seq

# Change length of sentences based on how long we've been training
lengthSchedule = {0: args.dataSentenceLength,
                  (args.epochs//3): args.dataSentenceLength + 10, 
                  (args.epochs//2): args.dataSentenceLength + 20, 
                  (args.epochs//1.5): args.dataSentenceLength + 30} 
lengthScheduling = False

'''
testData = 'data/de-en/train.tok.clean.bpe.32000.en'
targetData = 'data/de-en/train.tok.clean.bpe.32000.de'
testDataVal = 'data/de-en/newstest2011.tok.bpe.32000.en'
targetDataVal = 'data/de-en/newstest2011.tok.bpe.32000.de'
'''
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

trainingData = dataUtils.loadTrainingData(args.size, args.dataSentenceLength, testDataFile, targetDataFile, testLang, targetLang)
#testData = dataUtils.loadTestData(50, args.dataSentenceLength, testDataValFile, targetDataValFile, testLang.name, targetLang.name)
testData = dataUtils.loadTestData(40, args.dataSentenceLength, testDataFile, targetDataFile, testLang.name, targetLang.name)

dataLoader = torch.utils.data.DataLoader(trainingData, shuffle = True, num_workers = 4, 
                                         batch_size = args.batch, drop_last = True)

cuda = False
if torch.cuda.is_available():
    device = torch.device('cuda')
    cuda = True
    print('PROCESSING WITH CUDA DEVICE ', device)
else:
    device = torch.device('cpu')
    print('PROCESSING WITH CPU')
print('\n')

encoder = seq2seq.encoder(testLang.nWords, hiddenSize=args.hSize, numLayers = args.layers).to(device)
decoder = seq2seq.bahdanauDecoder(targetLang.nWords, hiddenSize=args.hSize, 
                              maxLength=args.maxWords, numLayers = args.layers).to(device)

loss_fn = nn.CrossEntropyLoss(ignore_index = testLang.PAD, reduction = 'sum')
encoderOptim = torch.optim.Adam(encoder.parameters(), lr= args.learningRate)
decoderOptim = torch.optim.Adam(decoder.parameters(), lr= args.learningRate)

teacherForceRatio = .5

startTime = datetime.datetime.now()
for epoch in range(args.epochs):
    epochTime = datetime.datetime.now()
    epochLoss = 0
    if lengthScheduling:
        if epoch in lengthSchedule.keys():
            print("Increasing sentence length")
            args.dataSentenceLength = lengthSchedule[epoch]
            trainingData = dataUtils.loadTrainingData(args.size, args.dataSentenceLength, 
                                                      testDataFile, targetDataFile, testLang, targetLang)
            dataLoader = torch.utils.data.DataLoader(trainingData, shuffle = True, 
                                                     num_workers = 0, batch_size = args.batch)
            print("Created new dataset")
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
        epochLoss += loss

        stepTime = datetime.datetime.now() - stepStartTime
    epochLoss = epochLoss.item() / args.size
    epochTime = datetime.datetime.now() - epochTime
    bleu = evaluateSeq2Seq.testBLEU(testData, encoder, decoder, testLang, targetLang, False)

    print(f"Epoch: {epoch+1}\tLoss: {round(epochLoss, 8)}\tEpoch Time: {epochTime}\tStep Time: {stepTime}\tBLEU: {bleu*100}")
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


evaluateSeq2Seq.testBLEU(testData, encoder, decoder, testLang, targetLang, True)
print('Final loss: \t', epochLoss)
print('Elapsed time: \t', elapsedTime)
