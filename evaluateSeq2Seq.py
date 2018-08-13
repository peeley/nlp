import torch, seq2seq, langModel, dataUtils
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu

def evaluate(encoder, decoder, rawString, testLang, targetLang, testTarget = None):
    cuda = False
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        encoder.to(device)
        decoder.to(device)
        cuda = True
    with torch.no_grad():
        for item in range(len(rawString)):
            inputString = langModel.expandContractions(langModel.normalize(rawString[item]))
            print('\nTest sentence: \t', inputString)
            inputSentence = langModel.tensorFromSentence(testLang, inputString, False)
            if cuda:
                inputSentence = inputSentence.cuda()
            if inputSentence.shape[0] == 1:
                if inputSentence.item() == -1:
                    return ['NONE']

            inputLength = len(inputSentence)
            encoderHidden = seq2seq.initHidden(cuda, encoder.hiddenSize, decoder.numLayers)
            encoderOutputs = torch.zeros(decoder.maxLength, encoder.hiddenSize * 2)
            if cuda:
                encoderOutpus = encoderOutputs.to(device)
            for word in range(inputLength):
                encoderOutput, encoderHidden = encoder(inputSentence[word], encoderHidden)
                encoderOutputs[word] = encoderOutput[0,0]

            decoderInput = torch.tensor([[targetLang.SOS]])
            if cuda:
                decoderInput = decoderInput.to(device)
            decoderHidden = encoderHidden
            decodedWords = []
            
            for letter in range(decoder.maxLength):
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
            print('Target: ', targetLine)
            print('BLEU Score: \t', bleu)
            bleuScores.append(bleu)
            bleuAVG = (sum(bleuScores)/len(bleuScores)) * 100
            print('BLEU Average: \t', bleuAVG, '\n')


if __name__ == '__main__':
    corpus, eng, de = dataUtils.loadEnDe(1000, 15)
    vocab = 5000 
    maxWords = 10

    while True:
        testString = input('Enter text to be translated: ')
        testData = [testString]
        savedEncoder = torch.load('encoder.pt')
        savedDecoder = torch.load('decoder.pt')
        translated = evaluate(savedEncoder, savedDecoder, testData, eng, de)
        printableTranslated = ' '.join(translated).encode('utf8')

