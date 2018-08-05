import torch, seq2seq, langModel, dataUtils
import pandas as pd

corpus, eng, ipq = dataUtils.loadDicts()
vocab = 5000 
maxWords = 10

encoder = torch.load('encoder.pt')
decoder = torch.load('decoder.pt')

train = False
cuda = False
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    encoder.to(device)
    decoder.to(device)
    cuda = True

def evaluate(rawString, testTarget = None):
    with torch.no_grad():
        for item in range(len(rawString)):
            inputString = langModel.expandContractions(langModel.normalize(rawString[item]))
            print('\nTest sentence: \t', inputString)
            inputSentence = langModel.tensorFromSentence(eng, inputString, train)
            if cuda:
                inputSentence = inputSentence.cuda()
            if inputSentence.shape[0] == 1:
                if inputSentence.item() == -1:
                    break
            
            inputLength = len(inputSentence)
            encoderHidden = seq2seq.initHidden(cuda, encoder.hiddenSize, decoder.numLayers)
            encoderOutputs = torch.zeros(maxWords, encoder.hiddenSize * 2)
            if cuda:
                encoderOutpus = encoderOutputs.to(device)
            for word in range(inputLength):
                encoderOutput, encoderHidden = encoder(inputSentence[word], encoderHidden)
                encoderOutputs[word] = encoderOutput[0,0]

            decoderInput = torch.tensor([[ipq.SOS]])
            if cuda:
                decoderInput = decoderInput.to(device)
            decoderHidden = encoderHidden
            decodedWords = []
            
            for letter in range(maxWords):
                decoderOutput, decoderHidden = decoder(decoderInput, decoderHidden, encoderOutputs)
                topv, topi = decoderOutput.data.topk(1)
                if topi.item() == 1:
                    break
                else:
                    decodedWords.append(ipq.idx2word[topi.item()])
                decoderInput = topi.squeeze().detach()
            print('Translated: \t', ' '.join(decodedWords),'\n')

while True:
    testString = input('Enter text to be translated: ')
    testData = [testString]
    evaluate(testData)

