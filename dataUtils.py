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

def loadEnglishSpanish():
    corpus = pd.read_csv('data/spa-eng/spa.txt', sep = '\t', lineterminator = '\n', names = ['eng','spa'])
    eng = langModel.langModel('english')
    spa = langModel.langModel('spanish')

    for row in range(corpus.shape[0]):
        eng.addSentence(langModel.normalize(corpus.iloc[row]['eng']))
        spa.addSentence(langModel.normalize(corpus.iloc[row]['spa']))
    return corpus, eng, spa

def loadInupiaqBible():
    print('Loading parallel texts...')
    data = pd.read_csv('data/inupiaq/bible.csv', names = ['eng', 'ipq'])
    eng = langModel.langModel('english')
    ipq = langModel.langModel('inupiaq')
    print('Parallel texts loaded, constructing language...')
    for row in range(data.shape[0]):
        eng.addSentence(langModel.normalize(data.iloc[row]['eng']))
        ipq.addSentence(langModel.normalize(data.iloc[row]['ipq']))
    print('Language constructed.')
    return data, eng, ipq 

def loadIpqDicts():
    print('Creating new dataframe...')
    frame = pd.DataFrame(columns = ['eng', 'ipq'])
    index = 0
    eng = langModel.langModel('english')
    ipq = langModel.langModel('inupiaq')
    with open('data/inupiaq/seiler_ipq_bpe.txt') as ipqFile:
        for line in ipqFile:
            line = line.strip('\n')
            ipq.addSentence(langModel.normalize(line))
            frame.loc[index, 'ipq'] = line
            index += 1
    index = 0
    with open('data/inupiaq/seiler_eng_bpe.txt') as engFile:
        for line in engFile:
            line = line.strip('\n')
            eng.addSentence(langModel.normalize(line))
            frame.loc[index, 'eng'] = line
            index += 1
    frame.to_csv('data/inupiaq/maclean.csv')
    return frame, eng, ipq

def loadEnDe(vocabSize, words):
    print('Creating dataset...')
    frame = pd.DataFrame(columns = ['eng', 'de'])
    index = 0
    eng = langModel.langModel('eng')
    de = langModel.langModel('de')
    deFile = open('data/de-en/train.tok.clean.bpe.32000.de', encoding = 'utf8')
    engFile = open('data/de-en/train.tok.clean.bpe.32000.en', encoding = 'utf8')
    print('Creating language models...')
    for deLine, engLine in zip(deFile, engFile):
        if index == vocabSize:
            break
        deLine = deLine.strip('\n')
        engLine = engLine.strip('\n')
        de.addSentence(langModel.normalize(deLine))
        eng.addSentence(langModel.normalize(engLine))
        if len(deLine.split()) < words and len(engLine.split()) < words:
            frame.loc[index, 'de'] = deLine
            frame.loc[index, 'eng'] = engLine
            index += 1
    print('Creation complete, ', index, ' lines.')
    deFile.close()
    engFile.close()
    frame.to_csv('data/de-en/de-en.csv')
    dataset = LangDataset(frame, eng, de, words)
    return dataset, eng, de 

def loadTestEnDe(size):
    index = 0
    frame = pd.DataFrame(columns = ['eng', 'de'])
    deFile = open('data/de-en/newstest2012.tok.bpe.32000.de')
    engFile = open('data/de-en/newstest2012.tok.bpe.32000.en')
    print('Creating test dataset...')
    for deLine, engLine in zip(deFile, engFile):
        deLine = deLine.strip('\n')
        engLine = engLine.strip('\n')
        if len(deLine.split()) < size and len(engLine.split()) < size:
            frame.loc[index, 'de'] = deLine
            frame.loc[index, 'eng'] = engLine
            index += 1
    print('Creation complete.')
    deFile.close()
    engFile.close()
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

