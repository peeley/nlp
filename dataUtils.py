import pandas as pd
import unicodedata, string, torch

all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters) + 1

def constructJoke():
    jokesFrame = pd.read_json('data/jokes/reddit_jokes.json')
    jokesFrame['fullJoke'] = jokesFrame['title'].map(str) + ' ' + jokesFrame['body'] + '>'
    jokesFrame['fullJoke'] = [unicodeToAscii(line) for line in jokesFrame['fullJoke']]
    jokesFrame['fullJoke'] = jokesFrame['fullJoke'].str.lower()
    filter = (jokesFrame['fullJoke'].str.len() <= 50)
    jokesFrame = jokesFrame.loc[filter]
    print(jokesFrame['fullJoke'])
    return jokesFrame['fullJoke']

def unicodeToAscii(s):
    all_letters = string.ascii_letters + " .,;'-"
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

def letterToTensor(letter):
    alphabet = "abcdefghijklmnopqrstuvwxyz.'!? "
    letterIndex = alphabet.index(letter)
    oneHot = torch.zeros(len(alphabet))
    oneHot[letterIndex] = 1
    return oneHot

def tensorToLetter(tensor):
    alphabet = "abcdefghijklmnopqrstuvwxyz.'!? "
    oneIndex = torch.argmax(tensor)
    return oneIndex

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

