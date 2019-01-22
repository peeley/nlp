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


sourceDataTrainingFile = 'data/inupiaq/data_eng_train'
targetDataTrainingFile = 'data/inupiaq/data_ipq_train_bpe'
sourceDataValFile = 'data/inupiaq/data_eng_val'
targetDataValFile = 'data/inupiaq/data_ipq_val_bpe' 

try:
    print('Loading saved language models.')
    with open('src/models/source.p', 'rb') as sourcePickle:
        sourceLang = pickle.load(sourcePickle)
    with open('src/models/target.p', 'rb') as targetPickle:
        targetLang = pickle.load(targetPickle)
except:
    print('No language models found, creating new models.')
    sourceLang = langModel.LangModel('eng')
    targetLang = langModel.LangModel('ipq')
    langModel.constructModels('data/inupiaq/data_eng', 'data/inupiaq/data_ipq', sourceLang, targetLang)

trainingData = dataUtils.loadData(args.size, args.maxWords, sourceDataTrainingFile, 
        targetDataTrainingFile, sourceLang, targetLang)
testData = dataUtils.loadData(100, args.maxWords, sourceDataValFile, targetDataValFile, sourceLang, targetLang)
dataLoader = torch.utils.data.DataLoader(trainingData, shuffle = True, num_workers = 4, batch_size = args.batch)
testLoader = torch.utils.data.DataLoader(testData, shuffle = True, num_workers = 4, batch_size = 1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Processing with device {device}.\n')
teacherForceRatio = .5
checkpointInterval = 10
decoderLearnRatio = 3.0

encoder = seq2seq.encoder(sourceLang.nWords, hiddenSize=args.hSize, numLayers = args.layers).to(device)
decoder = seq2seq.bahdanauDecoder(targetLang.nWords, hiddenSize=args.hSize, 
                              maxLength=args.maxWords, numLayers = args.layers).to(device)
loss_fn = nn.CrossEntropyLoss(ignore_index = sourceLang.PAD)
encoderOptim = torch.optim.Adam(encoder.parameters(), lr= args.lr)
decoderOptim = torch.optim.Adam(decoder.parameters(), lr= args.lr * decoderLearnRatio)

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
        nn.utils.clip_grad_norm_(decoder.parameters(), 40)
        nn.utils.clip_grad_norm_(encoder.parameters(), 40)
        encoderOptim.step()
        decoderOptim.step()
        epochLoss += loss.item()

        stepTime = datetime.datetime.now() - stepStartTime
    epochLoss = epochLoss / len(trainingData)
    epochTime = datetime.datetime.now() - epochTime
    bleu = evaluateUtils.testBLEU(testLoader, encoder, decoder, sourceLang, targetLang, False)
    print(f"Epoch: {epoch+1}\tLoss: {epochLoss:.5f}\tEpoch Time: {epochTime}\tStep Time: {stepTime}\tBLEU: {bleu*100:.5f}")
    if (epoch + 1) % checkpointInterval == 0:
        print('Checkpoint, saving models.')
        torch.save(encoder, 'src/models/encoder.pt')
        torch.save(decoder, 'src/models/decoder.pt')

endTime = datetime.datetime.now()
elapsedTime = endTime - startTime
print('Writing models to disk...')
torch.save(encoder, 'src/models/encoder.pt')
torch.save(decoder, 'src/models/decoder.pt')
print('Models saved to disk.\n')

evaluateUtils.testBLEU(testLoader, encoder, decoder, sourceLang, targetLang, True)
print(f"Final loss: \t{epochLoss}")
print(f"Elapsed time: \t{elapsedTime}")
