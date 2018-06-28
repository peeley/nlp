import langModel, seq2seq, torch
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt

maxWords = 10
minWords = 5

corpus = pd.read_csv('/mnt/9C4269244269047C/Programming/nlp/data/spa-eng/spa.txt', sep = '\t', lineterminator = '\n', names = ['eng','spa'])

mask = ((corpus['eng'].str.split().apply(len) < maxWords) & (corpus['eng'].str.split().apply(len) > minWords) ) & ((corpus['spa'].str.split().apply(len) < maxWords) & (corpus['spa'].str.split().apply(len) > minWords))
corpus = corpus[mask]

trainingData = corpus[:5000]

eng = langModel.langModel('english')
spa = langModel.langModel('spanish')
for row in range(trainingData.shape[0]):
    eng.addSentence(trainingData.iloc[row]['eng'])
    spa.addSentence(trainingData.iloc[row]['spa'])

epochs = 1
loss_fn = nn.NLLLoss()

encoder = seq2seq.encoder(eng.nWords, hiddenSize = 256, lr = .01)
decoder = seq2seq.decoder(spa.nWords, hiddenSize = 256, lr = .01, dropoutProb = .1)

encoderOptim = torch.optim.SGD(encoder.parameters(), encoder.lr)
decoderOptim = torch.optim.SGD(decoder.parameters(), decoder.lr)
losses = []

for epoch in range(epochs):
    for row in range(trainingData.shape[0]):
        loss = 0
        input, target = langModel.tensorFromPair(eng, spa, trainingData.iloc[row]['eng'], trainingData.iloc[row]['spa'])
        encoderOptim.zero_grad()
        decoderOptim.zero_grad()

        encoderHidden = encoder.initHidden()
        encoderOutputs = torch.zeros(maxWords, encoder.hiddenSize)
        print('Encoding sentence...')
        for inputLetter in range(input.shape[0]):
            encoderOutput, encoderHidden = encoder(input[inputLetter], encoderHidden)
            encoderOutputs[inputLetter] = encoderOutput[0,0]
        
        decoderInput = torch.tensor([[0]])
        decoderHidden = encoderHidden

        print('Decoding sentence...')
        for targetLetter in range(target.shape[0]):
            decoderOutput, decoderHidden = decoder(decoderInput, decoderHidden)
            topv, topi = decoderOutput.topk(1)
            decoderInput = topi.squeeze().detach()
            loss += loss_fn(decoderOutput, target[targetLetter])
            if decoderInput.item() == 1:
                break

        loss.backward()
        encoderOptim.step()
        decoderOptim.step()
        losses.append(loss)
        print('Loss: ', loss.item(), '\n')

plt.plot(losses)
plt.show()
