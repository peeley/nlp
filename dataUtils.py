import pandas as pd
import unicodedata, string, torch, langModel, langModel, random
import torch.utils.data
from nltk import word_tokenize

all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters)

class LangDataset(torch.utils.data.Dataset):
    def __init__(self, frame, testLang, targetLang, length, transform = None):
        self.frame = frame
        self.testLang = testLang
        self.targetLang = targetLang
        self.length = length

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        testLine = langModel.normalize(self.frame.loc[idx, self.testLang.name])
        testLine = ' '.join(word_tokenize(testLine))
        targetLine = langModel.normalize(self.frame.loc[idx, self.targetLang.name])
        testTensor, targetTensor = langModel.tensorFromPair(self.testLang, self.targetLang, testLine, targetLine)
        return (testTensor, targetTensor, testLine, targetLine)

def loadTrainingData(vocabSize, words, testFilename, targetFilename, testLang, targetLang):
    print('Creating dataset...')
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
        targetLang.addSentence(langModel.normalize(targetLine))
        # Tokenize english with NLTK tokenizer
        testSent = ' '.join(word_tokenize(langModel.normalize(testLine)))
        testLang.addSentence(testSent)
        if len(targetLine.split()) < words and len(testLine.split()) < words:
            frame.loc[index, targetLang.name] = targetLine
            frame.loc[index, testLang.name] = testLine
            index += 1
    print('Creation complete, ', index, ' lines.')
    targetFile.close()
    testFile.close()
    frame.to_csv('data/inupiaq/data.csv')
    dataset = LangDataset(frame, testLang, targetLang, words)
    return dataset

def loadTestData(testFileName, targetFileName, testLang, targetLang):
    index = 0
    frame = pd.DataFrame(columns = [testLang.name, targetLang.name])
    targetFile = open(targetFileName)
    testFile = open(testFileName)
    print('Creating test dataset...')
    for testLine, targetLine in zip(testFile, targetFile):
        targetLine = targetLine.strip('\n')
        testLine = testLine.strip('\n')
        frame.loc[index, targetLang.name] = targetLine
        frame.loc[index, testLang.name] = testLine
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
