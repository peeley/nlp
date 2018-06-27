import langModel, seq2seq
import pandas as pd

maxWords = 10
minWords = 5

corpus = pd.read_csv('/mnt/9C4269244269047C/Programming/nlp/data/spa-eng/spa.txt', sep = '\t', lineterminator = '\n', names = ['eng','spa'])

mask = ((corpus['eng'].str.split().apply(len) < maxWords) & (corpus['eng'].str.split().apply(len) > minWords) ) & ((corpus['spa'].str.split().apply(len) < maxWords) & (corpus['spa'].str.split().apply(len) > minWords))
corpus = corpus[mask]


