import unicodedata, string, torch, random, nltk, torch.utils.data
from src import langModel, langModel
import torch.nn.utils.rnn as rnn
import pandas as pd

all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters)
nltk.download('punkt')

class LangDataset(torch.utils.data.Dataset):
    def __init__(self, frame, testLang, targetLang, length, transform = None):
        self.frame = frame
        self.testLang = testLang
        self.targetLang = targetLang
        self.length = length

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        testLine   = self.frame.loc[idx, self.testLang.name]
        targetLine = self.frame.loc[idx, self.targetLang.name]
        testTensor, targetTensor = langModel.tensorFromPair(self.testLang, self.targetLang, 
                                                            testLine, targetLine,
                                                            self.length)
        return (testTensor, targetTensor, testLine, targetLine)

def loadTrainingData(vocabSize, words, testFilename, targetFilename, testLang, targetLang):
    print('Creating dataset from file {}...'.format(testFilename))
    frame = pd.DataFrame(columns = [testLang.name, targetLang.name])
    index = 0
    testFile = open(testFilename, encoding = 'utf8')
    targetFile = open(targetFilename, encoding = 'utf8')
    print('Creating language models...')
    for testLine, targetLine in zip(testFile, targetFile):
        if index == vocabSize:
            break
        targetLine = targetLine.strip('\n')
        testLine = testLine.strip('\n')
        testLine = ' '.join(nltk.word_tokenize(langModel.normalize(testLine)))
        targetLine = ' '.join(nltk.word_tokenize(langModel.normalize(targetLine)))
        testLang.addSentence(testLine)
        targetLang.addSentence(targetLine)
        if len(targetLine.split()) < words and len(testLine.split()) < words:
            frame.loc[index, targetLang.name] = targetLine
            frame.loc[index, testLang.name] = testLine
            index += 1
    print(f'Creation complete, {index} lines.')
    print(f'Language {testLang.name} created with {testLang.nWords} words.')
    print(f'Language {targetLang.name} created with {targetLang.nWords} words.')
    targetFile.close()
    testFile.close()
    dataset = LangDataset(frame, testLang, targetLang, words)
    return dataset

def loadToyData(vocabSize, words, filename, testLang, targetLang):
    frame = pd.DataFrame(columns = [testLang.name, targetLang.name])
    index = 0
    with open(filename, encoding = 'utf8') as langFile:
        print('Creating training dataset from file {}...'.format(filename))
        for line in langFile:
            if index == vocabSize:
                break
            sentences = line.strip('\n').split('\t')
            testLine = ' '.join(nltk.word_tokenize(langModel.normalize(sentences[0])))
            targetLine = ' '.join(nltk.word_tokenize(langModel.normalize(sentences[1])))
            testLang.addSentence(testLine)
            targetLang.addSentence(targetLine)

            frame.loc[index, targetLang.name] = targetLine
            frame.loc[index, testLang.name] = testLine
            index += 1
    print(f'Training dataset created from {index} lines')
    dataset = LangDataset(frame, testLang, targetLang, words)
    return dataset

def loadToyTest(vocabSize, words, filename, testLang, targetLang):
    index = 0
    frame = pd.DataFrame(columns = [testLang, targetLang])
    with open(filename, encoding = 'utf8') as file:
        print('Creating test dataset from file {}...'.format(filename))
        for line in file:
            if index == vocabSize:
                break
            sentences = line.strip('\n').split('\t')
            testLine = ' '.join(nltk.word_tokenize(langModel.normalize(sentences[0])))
            targetLine = ' '.join(nltk.word_tokenize(langModel.normalize(sentences[1])))
            if len(targetLine.split()) < words and len(testLine.split()) < words:
                frame.loc[index, targetLang] = targetLine
                frame.loc[index, testLang] = testLine
                index += 1
    print('Creation complete.')
    return frame

def loadTestData(vocabSize, words, testFileName, targetFileName, testLang, targetLang):
    index = 0
    frame = pd.DataFrame(columns = [testLang, targetLang])
    targetFile = open(targetFileName)
    testFile = open(testFileName)
    print('Creating training dataset from file {}...'.format(testFileName))
    for testLine, targetLine in zip(testFile, targetFile):
        if index == vocabSize:
            break
        targetLine = targetLine.strip('\n')
        testLine = langModel.normalize(testLine.strip('\n'))
        if len(targetLine.split()) < words and len(testLine.split()) < words:
            frame.loc[index, targetLang] = targetLine
            frame.loc[index, testLang] = testLine
            index += 1
    print('Creation complete.')
    testFile.close()
    targetFile.close()
    return frame

def splitData(testFilename, targetFilename):
    engFile = open(testFilename, 'r')
    ipqFile = open(targetFilename, 'r')
    engDataFile = open(testFilename+'_train', 'w+')
    ipqDataFile = open(targetFilename+'_train', 'w+')
    engValFile = open(testFilename+'_val', 'w+')
    ipqValFile = open(targetFilename+'_val', 'w+')
    engTestFile = open(testFilename+'_test', 'w+')
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
            engDataFile.write(engLine)
            ipqDataFile.write(ipqLine)
    engFile.close()     ; ipqFile.close()
    engDataFile.close() ; ipqDataFile.close()
    engValFile.close()  ; ipqValFile.close()
    engTestFile.close() ; ipqTestFile.close()

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

if __name__ == '__main__':
    splitData('data/inupiaq/data_eng', 'data/inupiaq/data_ipq')
