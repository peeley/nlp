import torch, re, pickle
import numpy as np

class langModel:
    def __init__(self, name):
        self.name = name
        self.word2idx = {'/start/': 0, '/end/':1}
        self.word2count = {}
        self.idx2word = {0: '/start/', 1:'/end/'}
        self.idx2word = {}
        self.nWords = 2
        self.EOS = 1
        self.SOS = 0
        self.glove = []

    def addEmbedding(self, filepath, filename):
        print('Opening saved embedding...')
        try:
            word2idx = pickle.load(open(filepath+'word2idx.pkl', 'rb'))
            glove = pickle.load(open(filepath+'glove.pkl', 'rb'))
            self.word2idx = word2idx
            self.glove = glove
            print('Opened saved embedding.')
        except:
            with open(filepath+filename, 'rb') as f:
                print('No embedding found, creating new embedding.')
                for l in f:
                        line = l.decode().split()
                        word = line[0]
                        print('Parsing new word # {}: {}'.format(self.nWords, word))
                        if word not in self.word2idx:
                            self.nWords += 1
                            self.word2count[word] = 1
                            self.word2idx[word] = self.nWords
                            self.idx2word[self.nWords] = word
                            vect = np.array(line[1:]).astype(np.float)
                            self.glove.append(vect)
                            print('Word vector: ', vect, '\n')
                        else:
                            self.word2count[word] += 1
                            print('Duplicate word')
                pickle.dump(self.word2idx, open(filepath+'word2idx.pkl', 'wb'))
                pickle.dump(self.glove, open(filepath+'glove.pkl', 'wb'))
        print('Finished embedding!')
    
    def addSentence(self, string):
        for word in string.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.nWords
            self.word2count[word] = 1
            self.idx2word[self.nWords] = word
            self.nWords += 1
        else:
            self.word2count[word] += 1

def tensorFromSentence(lang, sentence):
    indices = []
    rareWords = {}
    for num, word in enumerate(sentence.split(' ')):
        try:
            indices.append(lang.word2idx[word])
        except KeyError as e:
            rareWords[num] = word
            print('WARNING - Word not in vocabulary: ', word)
    indices.append(1)
    indices = torch.tensor(indices, dtype = torch.long).view(-1,1)
    return indices, rareWords

def tensorFromPair(inputLang, outputLang, inputSentence, outputSentence):
    input = tensorFromSentence(inputLang, inputSentence)[0]
    target = tensorFromSentence(outputLang, outputSentence)[0]
    return input, target

def normalize(s):
    s = str(s)
    s = (s.lower().strip())
    s = re.sub(r"([,.!?])", r" \1", s)
    return s
