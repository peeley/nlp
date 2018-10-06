import torch, seq2seq, langModel, dataUtils, json, pickle
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu

def evaluate(encoder, decoder, rawString, testLang, targetLang):
    with open('params.json') as paramsFile:
        params = json.load(paramsFile)
    hSize    = params['hSize']
    maxWords = params['maxWords']
    layers   = params['layers']

    cuda = False
    if torch.cuda.is_available():
        device = torch.device('cuda')
        encoder.to(device)
        decoder.to(device)
        cuda = True
    else:
        device = torch.device('cpu')
        cuda = False
    with torch.no_grad():
        for item in range(len(rawString)):
            inputString = langModel.expandContractions(langModel.normalize(rawString[item]))
            print('\nTest sentence: \t', inputString)
            inputSentence = langModel.tensorFromSentence(testLang, inputString, False).view(-1,1).to(device)

            inputLength = len(inputSentence)
            encoderHidden = seq2seq.initHidden(cuda, hSize, layers * 2)
            encoderOutputs = torch.zeros(maxWords, hSize * 2).to(device)
            for word in range(inputSentence.shape[0]):
                encoderOutput, encoderHidden = encoder(inputSentence[word], encoderHidden)
                encoderOutputs[word] = encoderOutput[0,0]

            decoderInput = torch.tensor([[targetLang.SOS]]).to(device)
            decoderHidden = encoderHidden
            decodedWords = []
            
            for letter in range(maxWords):
                decoderOutput, decoderHidden = decoder(decoderInput, decoderHidden, encoderOutputs)
                topv, topi = decoderOutput.data.topk(1)
                if topi.item() == 1:
                    break
                else:
                    decodedWords.append(targetLang.idx2word[topi.item()])
                decoderInput = topi.squeeze().detach()
            print('Translated: \t', ' '.join(decodedWords))
            return decodedWords

def testBLEU(testData, encoder, decoder, testLang, targetLang):
    with torch.no_grad():
        bleuAVG = 0
        bleuScores = []
        print('--- TESTING BLEU SCORES ---')
        for index, line in testData.iterrows():
            testLine    = line[testLang.name]
            targetLine  = line[targetLang.name]
            decodedString = evaluate(encoder, decoder, [testLine], testLang, targetLang)
            if '' in decodedString:
                decodedString = list(filter(None, decodedString))
            if '/end/' in decodedString: 
                decodedString.remove('/end/')
            if decodedString == None or decodedString == -1:
                bleu = 0
            else:
                bleu = sentence_bleu([targetLine.split()], decodedString)
            print('Target: \t', targetLine)
            print('BLEU Score: \t', bleu)
            bleuScores.append(bleu)
            bleuAVG = (sum(bleuScores)/len(bleuScores)) * 100
            print('BLEU Average: \t', bleuAVG, '\n')
        print('\nFinal BLEU: \t', bleuAVG)


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('\nEvaluating with device ', device)
    print('Loading saved models...')
    savedEncoder = torch.load('encoder.pt', map_location = device)
    savedDecoder = torch.load('decoder.pt', map_location = device)
    print('Saved models loaded.')
    with open('params.json') as paramsFile:
        params = json.load(paramsFile)
    print('Loading language models...')
    with open('eng.p', 'rb') as engFile:
        eng = pickle.load(engFile)
    with open('ipq.p', 'rb') as ipqFile:
        ipq = pickle.load(ipqFile)
    print('Language models loaded.')

    while True:
        testString = input('Enter text to be translated: ')
        testData = [testString]
        translated = evaluate(savedEncoder, savedDecoder, testData, eng, ipq)
        printableTranslated = ' '.join(translated).encode('utf8')

