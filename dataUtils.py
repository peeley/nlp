import pandas as pd
import unicodedata, string, torch, langModel, langModel

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
        print('Encoding sentence: \t', testLine)
        targetLine = langModel.normalize(self.frame.loc[idx, self.targetLang.name])
        print('Target setnence: \t', targetLine)
        testTensor, targetTensor = langModel.tensorFromPair(self.testLang, self.targetLang, testLine, targetLine)
        return (testTensor, targetTensor)

def loadTrainingData(vocabSize, words, testFilename, targetFilename, testLangName, targetLangName):
    print('Creating dataset...')
    testLang = langModel.langModel(testLangName)
    targetLang = langModel.langModel(targetLangName)
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
        testLang.addSentence(langModel.normalize(testLine))
        if len(targetLine.split()) < words and len(testLine.split()) < words:
            frame.loc[index, targetLang.name] = targetLine
            frame.loc[index, testLang.name] = testLine
            index += 1
    print('Creation complete, ', index, ' lines.')
    targetFile.close()
    testFile.close()
    frame.to_csv('data/inupiaq/data.csv')
    dataset = LangDataset(frame, testLang, targetLang, words)
    return dataset, testLang, targetLang

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

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

