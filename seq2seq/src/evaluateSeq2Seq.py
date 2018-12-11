#!/usr/bin/python

import torch, json, pickle, sys
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from src import langModel, seq2seq, dataUtils

with open('src/models/params.json') as paramsFile:
    params = json.load(paramsFile)
hSize    = params['hSize']
maxWords = params['maxWords']
layers   = params['layers']
length   = params['dataSentenceLength']

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

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
                    decoderInput = torch.tensor([topi.squeeze().detach()]).to(device)
    return decodedWords

def testBLEU(testData, encoder, decoder, testLang, targetLang, verbose):
    with torch.no_grad():
        bleuAVG = 0
        bleuScores = []
        if verbose:
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
            if len(decodedString) >= 4 and verbose:
                print(f'\nItem: \t#{index}/{testData.shape[0]}')
                print('Test: \t\t', testLine)
                print('Translated: \t', ' '.join(decodedString))
                print('Target: \t', targetLine)
                print(f'BLEU Score: \t{bleu} ({bleu*100})')
                print('BLEU Average: \t{:.8} ({:.8})\n'.format(bleuAVG, bleuAVG * 100))
        if verbose:
            print('\nFinal BLEU:\t{:.8} ({:.8})\n'.format(bleuAVG, bleuAVG * 100))
    return bleuAVG


if __name__ == '__main__':
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
        testBLEU(testData, savedEncoder, savedDecoder, eng, ipq, True)
        exit()
    testString = input('\nEnter text to be translated: ')
    while testString != '':
        testData = [testString]
        decoded = evaluate(savedEncoder, savedDecoder, testData, eng, ipq)
        print('Translated: ', ' '.join(decoded))
        testString = input('\nEnter text to be translated: ')
    print('Goodbye!')
