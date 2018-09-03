import pandas as pd
import unicodedata, string, torch, langModel, os, langModel

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
        testTensor, targetTensor = langModel.tensorFromPair(self.testLang, self.targetLang, testLine, targetLine, self.length)
        return (testTensor, targetTensor)

def constructJoke():
    print('Importing data...')
    jokesFrame = pd.read_json('data/jokes/reddit_jokes.json')
    jokesFrame = jokesFrame.loc[jokesFrame['score'] > 1]
    jokesFrame['fullJoke'] = jokesFrame['title'].map(str) + ' ' + jokesFrame['body']

    stupidFrame = pd.read_json('data/jokes/stupidstuff.json')
    stupidFrame = stupidFrame.loc[stupidFrame['category'].str.contains('Joke') & stupidFrame['rating'] > 1]

    wockaFrame = pd.read_json('data/jokes/wocka.json')

    fullFrame = pd.DataFrame(pd.concat((jokesFrame['fullJoke'], stupidFrame['body'], wockaFrame['body'])))
    fullFrame[0] = [unicodeToAscii(line) for line in fullFrame[0]]
    filter = (fullFrame[0].str.len() <= 100)
    fullFrame = fullFrame.loc[filter]
    fullFrame[0] = fullFrame[0].str.lower()

    return fullFrame[0]

def constructTweets():
    tweetFrame = pd.read_csv('data/tweets/data_backup.csv', engine ='python')
    tweetFrame['Text'] = tweetFrame['Text'].str.lower()
    return tweetFrame

def constructPokemon():
    nameFrame = pd.read_csv('/mnt/9C4269244269047C/Programming/nlp/data/pokemon/Pokemon.csv')
    nameFrame['Name'] = nameFrame['Name'].astype(str) + '>'
    return nameFrame['Name']

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
    frame.to_csv('data/de-en/de-en.csv')
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

def inputTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    tensor.view(-1, n_letters)
    return tensor

def targetTensor(line):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1) # EOS
    return torch.LongTensor(letter_indexes)

def word2idx(data):
    word2idx = {}
    for row in data:
        sentence = row.split()
        for word in sentence:
            if word not in word2idx:
                word2idx[word] = len(word2idx)
    return word2idx

def prepareSequence(seq, toIX):
    idxs = [toIX[w] for w in seq]
    return torch.tensor(idxs, dtype = torch.long)

def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][all_letters.find(letter)] = 1
    return tensor

