
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.utils.data_utils import get_file

import numpy as np
import random
import sys
import os

path = "data/text_generation/new" #path to training data, preferably saved as a .txt file.

try: 
    text = open(path).read().lower()
except UnicodeDecodeError:
    import codecs
    text = codecs.open(path, encoding='utf-8').read().lower()

print('corpus length:', len(text))

chars = set(text)
words = set(open(path).read().lower().split())


print("Unique words",len(words))
print("Unique chars", len(chars))


word_indices = dict((c, i) for i, c in enumerate(words))
indices_word = dict((i, c) for i, c in enumerate(words))

maxlen = 50
step = 3
print("maxlen:",maxlen,"step:", step)
sentences = []
next_words = []
next_words= []
sentences1 = []
list_words = []

sentences2=[]
list_words=text.lower().split()


for i in range(0,len(list_words)-maxlen, step):
    sentences2 = ' '.join(list_words[i: i + maxlen])
    sentences.append(sentences2)
    next_words.append((list_words[i + maxlen]))
print('Length of sentences:', len(sentences))

print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(words)), dtype=np.bool)
y = np.zeros((len(sentences), len(words)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, word in enumerate(sentence.split()):
        X[i, t, word_indices[word]] = 1
    y[i, word_indices[next_words[i]]] = 1


#build the model: 2 stacked LSTM cells
model = Sequential()
model.add(LSTM(256, return_sequences=True, input_shape=(maxlen, len(words))))
model.add(Dropout(0.4))
model.add(LSTM(256, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(len(words)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

#Comment out the below lines if this is the first time you're training on new data.
if os.path.isfile('backup/response_models/TextGeneration'):
    model.load_model('backup/response_models/TextGeneration.h5')

def sample(a, temperature=0.8):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

# train the model, output generated text after each iteration
no_of_iterations = 20

for iteration in range(1, no_of_iterations):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X, y, batch_size=128, epochs=3)
    model.save('backup/response_models/Textweights.h5',overwrite=True)

    start_index = random.randint(0, len(list_words) - maxlen - 1)

    for diversity in [0.3,0.6, 1.3, 1.6, 1.9]:
        print()
        print('Diversity:', diversity)
        generated = ''
        sentence = list_words[start_index: start_index + maxlen]
        generated += ' '.join(sentence)
        print('Seed: "' , sentence , '"')
        print()
        sys.stdout.write(generated)
        print()

        for i in range(200):
            x = np.zeros((1, maxlen, len(words)))
            for t, word in enumerate(sentence):
                x[0, t, word_indices[word]] = 1.

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_word = indices_word[next_index]
            generated += next_word
            del sentence[0]
            sentence.append(next_word)
            sys.stdout.write(' ')
            sys.stdout.write(next_word)
            sys.stdout.flush()
        print()
