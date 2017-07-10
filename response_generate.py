#Test response generation model
#Input: Seed(sentence) from user
#Output: Response from bot - One response per diversity value.

from keras.models import load_model
import copy
import numpy as np
import os
import random

print("\n Test response generation model : \n\n")

path = "data/text_generation/dialogues_edit" #path to training data, preferably saved as a .txt file.

try: 
    text = open(path).read().lower()
except UnicodeDecodeError:
    import codecs
    text = codecs.open(path, encoding='utf-8').read().lower()

words = set(text.split())
print("\nVocabulary size: ",len(words))

word_indices = dict((c, i) for i, c in enumerate(words))
indices_word = dict((i, c) for i, c in enumerate(words))


model = model.load_model('backup/response_models/TextGeneration.h5')
print("Loaded response generation model.")
test_text = input("\nEnter seed: ")
seed = copy.copy(test_text)

def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
	a = np.log(a) / temperature
	dist = np.exp(a)/np.sum(np.exp(a))
	choices = range(len(a))
	return np.random.choice(choices, p=dist)

no_of_iterations = 2000


for diversity in [0.3,0.6, 1.3, 1.6, 1.9]:
        
        print('Diversity:', diversity)
        generated = ''
        print('Seed:  "' , seed , '"')
        sys.stdout.write(generated)
        print()

        for i in range(200):
            x = np.zeros((1, maxlen, len(words)))
            for t, word in enumerate(seed):
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

