import unicodedata, string, torch, random, nltk, torch.utils.data
from src import langModel, langModel
import torch.nn.utils.rnn as rnn
import pandas as pd

all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters)
nltk.download('punkt')

class LangDataset(torch.utils.data.Dataset):
    def __init__(self, frame, sourceLang, targetLang, length):
        self.frame = frame
        self.sourceLang = sourceLang
        self.targetLang = targetLang
        self.length = length

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        sourceLine   = self.frame.loc[idx, self.sourceLang.name]
        targetLine = self.frame.loc[idx, self.targetLang.name]
        testTensor, targetTensor = langModel.tensorFromPair(self.sourceLang, self.targetLang, 
                                                            sourceLine, targetLine,
                                                            self.length)
        return (testTensor, targetTensor, sourceLine, targetLine)

# take in two files corresponding to aligned translations, create language models and dataset
# vocabSize : number of lines to include in dataset
# words : max number of tokens in each line
# sourceFilename, targetFilename : filenames of source and target language translations
# sourceLang, targetLang : language models to construct as defined in langModel
# train : boolean, true if creating training dataset and false otherwise
def loadData(vocabSize, words, sourceFilename, targetFilename, sourceLang, targetLang, train):
    print('Creating dataset from file {}...'.format(sourceFilename))
    index = 0
    frame = pd.DataFrame(columns = [sourceLang.name, targetLang.name])
    sourceFile = open(sourceFilename, encoding = 'utf8')
    targetFile = open(targetFilename, encoding = 'utf8')
    for sourceLine, targetLine in zip(sourceFile, targetFile):
        if index == vocabSize:
            break
        targetLine = targetLine.strip('\n')
        sourceLine = sourceLine.strip('\n')
        sourceLine = ' '.join(nltk.word_tokenize(langModel.normalize(sourceLine)))
        targetLine = langModel.normalize(targetLine)
        if train:
            sourceLang.addSentence(sourceLine)
            targetLang.addSentence(targetLine)
        if len(targetLine.split()) < words and len(sourceLine.split()) < words:
            frame.loc[index, targetLang.name] = targetLine
            frame.loc[index, sourceLang.name] = sourceLine
            index += 1
    print(f'Creation complete, {index} lines.')
    if train:
        print(f'Language {sourceLang.name} created with {sourceLang.nWords} words.')
        print(f'Language {targetLang.name} created with {targetLang.nWords} words.')
    targetFile.close()
    sourceFile.close()
    dataset = LangDataset(frame, sourceLang, targetLang, words)
    return dataset

def loadToyData(vocabSize, words, filename, sourceLang, targetLang):
    frame = pd.DataFrame(columns = [sourceLang.name, targetLang.name])
    index = 0
    with open(filename, encoding = 'utf8') as langFile:
        print('Creating training dataset from file {}...'.format(filename))
        for line in langFile:
            if index == vocabSize:
                break
            sentences = line.strip('\n').split('\t')
            sourceLine = ' '.join(nltk.word_tokenize(langModel.normalize(sentences[0])))
            targetLine = ' '.join(nltk.word_tokenize(langModel.normalize(sentences[1])))
            sourceLang.addSentence(sourceLine)
            targetLang.addSentence(targetLine)

            frame.loc[index, targetLang.name] = targetLine
            frame.loc[index, sourceLang.name] = sourceLine
            index += 1
    print(f'Training dataset created from {index} lines')
    dataset = LangDataset(frame, sourceLang, targetLang, words)
    return dataset

def loadToyTest(vocabSize, words, filename, sourceLang, targetLang):
    index = 0
    frame = pd.DataFrame(columns = [sourceLang, targetLang])
    with open(filename, encoding = 'utf8') as file:
        print('Creating test dataset from file {}...'.format(filename))
        for line in file:
            if index == vocabSize:
                break
            sentences = line.strip('\n').split('\t')
            sourceLine = ' '.join(nltk.word_tokenize(langModel.normalize(sentences[0])))
            targetLine = ' '.join(nltk.word_tokenize(langModel.normalize(sentences[1])))
            if len(targetLine.split()) < words and len(sourceLine.split()) < words:
                frame.loc[index, targetLang] = targetLine
                frame.loc[index, sourceLang] = sourceLine
                index += 1
    print('Creation complete.')
    return frame

def splitData(sourceFilename, targetFilename):
    engFile = open(sourceFilename, 'r')
    ipqFile = open(targetFilename, 'r')
    engTrainingFile = open(sourceFilename+'_train', 'w+')
    ipqTrainingFile = open(targetFilename+'_train', 'w+')
    engValFile = open(sourceFilename+'_val', 'w+')
    ipqValFile = open(targetFilename+'_val', 'w+')
    engTestFile = open(sourceFilename+'_test', 'w+')
    ipqTestFile = open(targetFilename+'_test', 'w+')

    for engLine, ipqLine in zip(engFile, ipqFile):
        prob = random.random()
        if prob < .03:
            engTestFile.write(engLine)
            ipqTestFile.write(ipqLine)
        elif prob > .03 and prob < .06:
            engValFile.write(engLine)
            ipqValFile.write(ipqLine)
        else:
            engTrainingFile.write(engLine)
            ipqTrainingFile.write(ipqLine)
    engFile.close()     ; ipqFile.close()
    engTrainingFile.close() ; ipqTrainingFile.close()
    engValFile.close()  ; ipqValFile.close()
    engTestFile.close() ; ipqTestFile.close()
'''
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )
'''
if __name__ == '__main__':
    splitData('data/inupiaq/data_eng', 'data/inupiaq/data_ipq')
