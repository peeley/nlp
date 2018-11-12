#!/usr/bin/python

import torch, seq2seq, langModel, dataUtils, json, pickle, sys
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

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
            inputSentence, rareWords = langModel.tensorFromSentence(testLang, inputString, length)
            inputSentence = inputSentence.view(-1,1,1).to(device)

            encoderOutputs, encoderHidden = encoder(inputSentence, None)
            decoderInput = torch.tensor([[targetLang.SOS]]).to(device)
            decoderHidden = encoderHidden[:layers]
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
            decodedString = evaluate(encoder, decoder, [testLine], testLang, targetLang)
            if '' in decodedString:
                decodedString = list(filter(None, decodedString))
            if '/end/' in decodedString: 
                decodedString.remove('/end/')
            if decodedString == None or decodedString == -1:
                bleu = 0
            else:
                bleu = sentence_bleu([targetLine.split()], decodedString) 
            bleuScores.append(bleu)
            bleuAVG = (sum(bleuScores)/len(bleuScores))
            if len(decodedString) >= 4:
                print('\nItem: \t#{}/{}'.format(index, testData.shape[0]))
                print('Test: \t', testLine)
                print('Translated: \t', ' '.join(decodedString))
                print('Target: \t', targetLine)
                print('BLEU Score: \t', bleu)
                print('BLEU Average: \t{:.8} ({:.8})\n'.format(bleuAVG, bleuAVG * 100))
        print('\nFinal BLEU:\t{:.8} ({:.8})\n'.format(bleuAVG, bleuAVG * 100))


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
    if sys.argv[1] == 'test':
        filename = 'data/de-en/deu-eng/deu.txt'
        testData = dataUtils.loadToyTest(500, params['dataSentenceLength'], 
                                         filename, 'eng', 'ipq')
        testBLEU(testData, savedEncoder, savedDecoder, eng, ipq)
        exit()
    testString = input('\nEnter text to be translated: ')
    while testString != '':
        testData = [testString]
        decoded = evaluate(savedEncoder, savedDecoder, testData, eng, ipq)
        print('Translated: ', ' '.join(decoded))
        testString = input('\nEnter text to be translated: ')
    print('Goodbye!')
