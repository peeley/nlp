#!/usr/bin/python

import torch, seq2seq, langModel, dataUtils, json, pickle
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu

with open('params.json') as paramsFile:
    params = json.load(paramsFile)
hSize    = params['hSize']
maxWords = params['maxWords']
layers   = params['layers']
length   = params['dataSentenceLength']

if torch.cuda.is_available():
    device = torch.device('cuda')
    cuda = True
else:
    device = torch.device('cpu')
    cuda = False

def evaluate(encoder, decoder, rawString, testLang, targetLang, train = False):
    with torch.no_grad():
        for item in range(len(rawString)):
            inputString = (rawString[item])
            print('\nTest sentence: \t', inputString)
            inputSentence, rareWords = langModel.tensorFromSentence(testLang, inputString, length)
            inputSentence = inputSentence.view(1,-1,1).to(device)

            encoderHidden = seq2seq.initHidden(cuda, hSize, layers * 2)
            encoderOutputs = torch.zeros(1, maxWords, hSize * 2).to(device)
            for word in range(inputSentence.shape[0]):
                encoderOutput, encoderHidden = encoder(inputSentence[0, word], encoderHidden)
                encoderOutputs[0, word] = encoderOutput[0, 0]

            decoderInput = torch.tensor([[targetLang.SOS]]).to(device)
            decoderHidden = encoderHidden
            decodedWords = []
            
            for letter in range(maxWords):
                if letter in rareWords.keys():
                    decodedWords.append(rareWords[letter])
                else:
                    decoderOutput, decoderHidden = decoder(decoderInput, decoderHidden, encoderOutputs)
                    decoderOutput = decoderOutput.view(1, -1)
                    topv, topi = decoderOutput.data.topk(1)
                    if topi.item() == testLang.EOS:
                        decodedWords.append('/end/')
                        break
                    else:
                        decodedWords.append(targetLang.idx2word[topi.item()])
                    decoderInput = torch.tensor([topi.squeeze().detach()])
            print('Translated: \t', ' '.join(decodedWords))
    return decodedWords

def testBLEU(testData, encoder, decoder, testLang, targetLang):
    encoder.to(device)
    decoder.to(device)
    with torch.no_grad():
        bleuAVG = 0
        bleuScores = []
        print('--- TESTING BLEU SCORES ---')
        for index, line in testData.iterrows():
            testLine    = line[testLang.name]
            targetLine  = langModel.normalize(line[targetLang.name])
            print('Item: \t#{}/{}'.format(index, testData.shape[0]))
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
    print('\nEvaluating with device: ', device)
    print('Loading saved models...')
    savedEncoder = torch.load('encoder.pt', map_location = device)
    savedDecoder = torch.load('decoder.pt', map_location = device)
    print('Saved models loaded.')
    print('Loading language models...')
    with open('eng.p', 'rb') as engFile:
        eng = pickle.load(engFile)
    with open('ipq.p', 'rb') as ipqFile:
        ipq = pickle.load(ipqFile)
    print('Language models loaded.')
    while True:
        testString = input('\nEnter text to be translated: ')
        testData = [testString]
        evaluate(savedEncoder, savedDecoder, testData, eng, ipq)

