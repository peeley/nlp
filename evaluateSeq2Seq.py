import torch, seq2seq, langModel
import pandas as pd

corpus = pd.read_csv('data/spa-eng/spa.txt', sep = '\t', lineterminator = '\n', names = ['eng','spa'])
trainingData = corpus.iloc[:10000]

eng = langModel.langModel('english')
spa = langModel.langModel('spanish')

for row in range(trainingData.shape[0]):
    eng.addSentence(langModel.normalize(corpus.loc[row,'eng']))
    spa.addSentence(langModel.normalize(corpus.loc[row,'spa']))

encoder = seq2seq.encoder(eng.nWords+1, hiddenSize = 1024, lr = .01)
decoder = seq2seq.decoder(spa.nWords+1, hiddenSize = 1024, lr = .01, dropoutProb = .1)
encoder.load_state_dict(torch.load('encoder.pt'))
decoder.load_state_dict(torch.load('decoder.pt'))

train = False
cuda = True
if torch.cuda.is_available:
    device = torch.device('cuda:0')
    encoder.to(device)
    decoder.to(device)

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
            encoderHidden = encoder.initHidden(cuda)
            encoderOutputs = torch.zeros(inputLength, encoder.hiddenSize)
            if cuda:
                encoderOutpus = encoderOutputs.to(device)
            for word in range(inputLength):
                encoderOutput, encoderHidden = encoder(inputSentence[word], encoderHidden)
                encoderOutputs[word] = encoderOutput[0,0]

            decoderInput = torch.tensor([[0]])
            if cuda:
                decoderInput = decoderInput.to(device)
            decoderHidden = encoderHidden
            decodedWords = []
            
            for letter in range(inputLength):
                decoderOutput, decoderHidden = decoder(decoderInput, decoderHidden)
                topv, topi = decoderOutput.data.topk(1)
                if topi.item() == 1:
                    break
                else:
                    decodedWords.append(spa.idx2word[topi.item()])
                decoderInput = topi.squeeze().detach()
            if testTarget:
                print('Target: \t', testTarget)
            print('Translated: \t', ' '.join(decodedWords),'\n')

while True:
    testString = input('Enter text to be translated: ')
    testData = [testString]
    evaluate(testData)

