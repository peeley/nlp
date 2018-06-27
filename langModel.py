class langModel:
    def __init__(self, name):
        self.name = name
        self.word2idx = {}
        self.word2count = {}
        self.idx2word = {0: 'SOS', 1:'EOS'}
        self.nWords = 2
        self.EOS = 1
        self.SOS = 0

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

def idxFromSentence(lang, sentence):
    return [lang.word2idx[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    idx = idxFromSentence(sentence)
    idx.append(lang.EOS)
    return torch.tensor(idx, dtype = torch.long).view(-1,1)

def tensorFromPair(inputLang, outputLang, inputSentence, targetSentence):
    input = tensorFromSentence(inputLang, inputSentence)
    target = tensorFromSentence(outputLang, outputSentence)
    return input, target
