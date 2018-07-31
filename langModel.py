import torch, re, pickle
import numpy as np

class langModel:
    def __init__(self, name):
        self.name = name
        self.word2idx = {}
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

def idxFromSentence(lang, sentence, train):
    indices = []
    for word in sentence.split(' '):
        try:
            indices.append(lang.word2idx[word])
        except KeyError as e:
            if train:
                indices.append(lang.nWords)
            else:
                print('ERROR - Word not in vocabulary: ', e, '\n')
                return -1
    return indices

def tensorFromSentence(lang, sentence, train):
    idx = idxFromSentence(lang, sentence, train)
    if idx == -1:
        return torch.Tensor(-1)
    idx.append(lang.EOS)
    return torch.tensor(idx, dtype = torch.long).view(-1,1)

def tensorFromPair(inputLang, outputLang, inputSentence, outputSentence, train):
    input = tensorFromSentence(inputLang, inputSentence, train)
    target = tensorFromSentence(outputLang, outputSentence, train)
    return input, target

def normalize(s):
    s = str(s)
    s = (s.lower().strip())
    s = re.sub(r"([,.!?])", r" \1", s)
    return s

cList = {
  "ain't": "am not",
  "aren't": "are not",
  "can't": "cannot",
  "can't've": "cannot have",
  "'cause": "because",
  "could've": "could have",
  "couldn't": "could not",
  "couldn't've": "could not have",
  "didn't": "did not",
  "doesn't": "does not",
  "don't": "do not",
  "hadn't": "had not",
  "hadn't've": "had not have",
  "hasn't": "has not",
  "haven't": "have not",
  "he'd": "he would",
  "he'd've": "he would have",
  "he'll": "he will",
  "he'll've": "he will have",
  "he's": "he is",
  "how'd": "how did",
  "how'd'y": "how do you",
  "how'll": "how will",
  "how's": "how is",
  "i'd": "i would",
  "i'd've": "i would have",
  "i'll": "i will",
  "i'll've": "i will have",
  "i'm": "i am",
  "i've": "i have",
  "isn't": "is not",
  "it'd": "it had",
  "it'd've": "it would have",
  "it'll": "it will",
  "it'll've": "it will have",
  "it's": "it is",
  "let's": "let us",
  "ma'am": "madam",
  "mayn't": "may not",
  "might've": "might have",
  "mightn't": "might not",
  "mightn't've": "might not have",
  "must've": "must have",
  "mustn't": "must not",
  "mustn't've": "must not have",
  "needn't": "need not",
  "needn't've": "need not have",
  "o'clock": "of the clock",
  "oughtn't": "ought not",
  "oughtn't've": "ought not have",
  "shan't": "shall not",
  "sha'n't": "shall not",
  "shan't've": "shall not have",
  "she'd": "she would",
  "she'd've": "she would have",
  "she'll": "she will",
  "she'll've": "she will have",
  "she's": "she is",
  "should've": "should have",
  "shouldn't": "should not",
  "shouldn't've": "should not have",
  "so've": "so have",
  "so's": "so is",
  "that'd": "that would",
  "that'd've": "that would have",
  "that's": "that is",
  "there'd": "there had",
  "there'd've": "there would have",
  "there's": "there is",
  "they'd": "they would",
  "they'd've": "they would have",
  "they'll": "they will",
  "they'll've": "they will have",
  "they're": "they are",
  "they've": "they have",
  "to've": "to have",
  "wasn't": "was not",
  "we'd": "we had",
  "we'd've": "we would have",
  "we'll": "we will",
  "we'll've": "we will have",
  "we're": "we are",
  "we've": "we have",
  "weren't": "were not",
  "what'll": "what will",
  "what'll've": "what will have",
  "what're": "what are",
  "what's": "what is",
  "what've": "what have",
  "when's": "when is",
  "when've": "when have",
  "where'd": "where did",
  "where's": "where is",
  "where've": "where have",
  "who'll": "who will",
  "who'll've": "who will have",
  "who's": "who is",
  "who've": "who have",
  "why's": "why is",
  "why've": "why have",
  "will've": "will have",
  "won't": "will not",
  "won't've": "will not have",
  "would've": "would have",
  "wouldn't": "would not",
  "wouldn't've": "would not have",
  "y'all": "you all",
  "y'alls": "you alls",
  "y'all'd": "you all would",
  "y'all'd've": "you all would have",
  "y'all're": "you all are",
  "y'all've": "you all have",
  "you'd": "you had",
  "you'd've": "you would have",
  "you'll": "you you will",
  "you'll've": "you you will have",
  "you're": "you are",
  "you've": "you have"
}

c_re = re.compile('(%s)' % '|'.join(cList.keys()))

def expandContractions(text, c_re=c_re):
    def replace(match):
        return cList[match.group(0)]
    string = c_re.sub(replace, text).replace("'s", " is")
    string = string.replace("'ll", " will")
    return string

