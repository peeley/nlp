#!/usr/bin/python

import torch, json, pickle, sys
from nltk.translate.bleu_score import sentence_bleu
from src import langModel, seq2seq, dataUtils

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# takes in raw string and of source language and returns array of words translated to target language
# encoder, decoder : encoder and decoder objects as defined in seq2seq
# inputTensor : PyTorch tensor of word indices or raw string to be converted to tensor
# sourceLang, targetLang : language models as defined in langModel
def evaluate(encoder, decoder, inputTensor, sourceLang, targetLang):
    rareWords = {}
    layers = encoder.numLayers
    maxWords = decoder.maxLength
    with torch.no_grad():
        if type(inputTensor) == str:
            inputTensor, rareWords = langModel.tensorFromSentence(sourceLang, inputTensor, maxWords)
        inputSentence = inputTensor.view(-1,1,1).to(device)

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
                if topi.item() == sourceLang.EOS:
                    decodedWords.append('/end/')
                    break
                else:
                    decodedWords.append(targetLang.idx2word[topi.item()])
                decoderInput = torch.tensor([topi.squeeze()]).to(device)
    return decodedWords

# tests encoder/decoder accuracy with BLEU metric
# testData : torch DataLoader object
# encoder, decoder : self-written encoder/decoder objects as defined in seq2seq
# sourceLang, targetLang : LangModels of source and target languages
# verbose : boolean, allows for detailed print messages
def testBLEU(testData, encoder, decoder, sourceLang, targetLang, verbose):
    with torch.no_grad():
        bleuAVG = 0
        bleuScores = []
        if verbose:
            print('--- TESTING BLEU SCORES ---')
        for index, data in enumerate(testData):
            inputTensor, targetTensor = data[0].transpose(0,1).to(device), data[1].transpose(0,1).to(device)
            sourceLine, targetLine = data[2][0], data[3][0]
            decodedString = evaluate(encoder, decoder, inputTensor, sourceLang, targetLang)
            if '' in decodedString:
                decodedString = list(filter(None, decodedString))
            if '/end/' in decodedString: 
                decodedString.remove('/end/')
            if decodedString == None:
                bleu = 0
            else:
                bleu = sentence_bleu([targetLine.split()], decodedString) 
            bleuScores.append(bleu)
            bleuAVG = sum(bleuScores)/len(bleuScores)
            if len(decodedString) >= 4 and verbose:
                print(f'\nItem: \t#{index}/{len(testData)}')
                print(f'Test: \t\t{sourceLine}')
                print('Translated:\t', ' '.join(decodedString))
                print(f'Target: \t{targetLine}')
                print(f'BLEU Score: \t{bleu} ({bleu*100:.8f})')
                print(f'BLEU Average: \t{bleuAVG:.8f} ({(bleuAVG*100):.8f})\n')
        if verbose:
            print(f'\nFinal BLEU:\t{bleuAVG:.8f} ({(bleuAVG*100):.8f})\n')
    return bleuAVG
