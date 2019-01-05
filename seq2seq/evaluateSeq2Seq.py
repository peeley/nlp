#!/usr/bin/python

import torch, pickle
from src import evaluateUtils, dataUtils

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('\nEvaluating with device: ', device)
    print('Loading saved models...')
    savedEncoder = torch.load('src/models/encoder.pt', map_location = device)
    savedDecoder = torch.load('src/models/decoder.pt', map_location = device)
    print('Saved models loaded.')
    print('Loading language models...')
    with open('src/models/eng.p', 'rb') as engFile:
        eng = pickle.load(engFile)
    with open('src/models/ipq.p', 'rb') as ipqFile:
        ipq = pickle.load(ipqFile)
    print('Language models loaded.')
    testString = input('\nEnter text to be translated: ')
    while testString != '':
        decoded = evaluateUtils.evaluate(savedEncoder, savedDecoder, testString, eng, ipq)
        print('Translated: ', ' '.join(decoded))
        testString = input('\nEnter text to be translated: ')
    print('Goodbye!')
