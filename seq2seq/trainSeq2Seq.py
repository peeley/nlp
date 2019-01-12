#!/usr/bin/python

from src import langModel, seq2seq, dataUtils, evaluateUtils
import torch, random, datetime, json, pickle, argparse
import torch.nn as nn

parser = argparse.ArgumentParser(description = "Script to train Qalgu translator.")
parser.add_argument('--size', type=int, default = 10000, help="Default: 10000")
parser.add_argument('--epochs', type=int, default = 15, help="Default: 15")
parser.add_argument('--maxWords', type=int, default = 25, help="Default: 50")
parser.add_argument('--hSize', type=int, default = 256, help="Default: 1024")
parser.add_argument('--layers', type=int, default = 4, help="Default: 4")
parser.add_argument('--batch', type=int, default = 64, help="Default: 1")
parser.add_argument('--lr', type=float, default=.001, help="Default: .001")
args = parser.parse_args()


sourceDataFile = 'data/inupiaq/data_eng_train'
targetDataFile = 'data/inupiaq/data_ipq_train_bpe'
sourceDataValFile = 'data/inupiaq/data_eng_val'
targetDataValFile = 'data/inupiaq/data_ipq_val_bpe' 

sourceLang = langModel.langModel('eng')
targetLang = langModel.langModel('ipq')

trainingData = dataUtils.loadData(args.size, args.maxWords, sourceDataFile, targetDataFile, sourceLang, targetLang, train=True)
testData = dataUtils.loadData(150, args.maxWords, sourceDataValFile, targetDataValFile, sourceLang, targetLang, train=False)
dataLoader = torch.utils.data.DataLoader(trainingData, shuffle = True, num_workers = 4, 
                                         batch_size = args.batch, drop_last = True)
testLoader = torch.utils.data.DataLoader(testData, shuffle = True, num_workers = 4, batch_size = 1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Processing with device {device}.\n')

encoder = seq2seq.encoder(sourceLang.nWords, hiddenSize=args.hSize, numLayers = args.layers).to(device)
decoder = seq2seq.bahdanauDecoder(targetLang.nWords, hiddenSize=args.hSize, 
                              maxLength=args.maxWords, numLayers = args.layers).to(device)
loss_fn = nn.CrossEntropyLoss(ignore_index = sourceLang.PAD, reduction = 'sum')
encoderOptim = torch.optim.Adam(encoder.parameters(), lr= args.lr)
decoderOptim = torch.optim.Adam(decoder.parameters(), lr= args.lr)

teacherForceRatio = .5
checkpointInterval = 10

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
        decoderInput = torch.LongTensor([sourceLang.SOS] * batchSize).to(device)
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
    bleu = evaluateUtils.testBLEU(testLoader, encoder, decoder, sourceLang, targetLang, False)
    print(f"Epoch: {epoch+1}\tLoss: {epochLoss:.5f}\tEpoch Time: {epochTime}\tStep Time: {stepTime}\tBLEU: {bleu*100:.5f}")
    if (epoch + 1) % checkpointInterval == 0:
        print('Checkpoint, saving models.')
        torch.save(encoder, 'src/models/encoder.pt')
        torch.save(decoder, 'src/models/decoder.pt')

endTime = datetime.datetime.now()
elapsedTime = endTime - startTime
# Save hyperparameters for loading models to production
settingsDict = {}
for arg in vars(args):
    value = getattr(args,arg)
    settingsDict[arg] = value
    print(f'Writing parameter {arg} = {value}')
with open('src/models/params.json', 'w+') as params:
    print('\nWriting parameters to disk...')
    json.dump(settingsDict, params)
    print('Saved parameters to disk.')
# Save language models
print('Writing language models to disk...')
with open(f"src/models/{sourceLang.name}.p", 'wb') as engFile:
    pickle.dump(sourceLang, engFile)
with open(f"src/models/{targetLang.name}.p", 'wb') as ipqFile:
    pickle.dump(targetLang, ipqFile)
print('Language models saved to disk.')
print('Writing models to disk...')
torch.save(encoder, 'src/models/encoder.pt')
torch.save(decoder, 'src/models/decoder.pt')
print('Models saved to disk.\n')

evaluateUtils.testBLEU(testLoader, encoder, decoder, sourceLang, targetLang, True)
print(f"Final loss: \t{epochLoss}")
print(f"Elapsed time: \t{elapsedTime}")
