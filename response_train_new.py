from preprocessor import Response_Dataset, get_sentence_vector, pad_vec_sequences2, vec_to_word
from keras.models import Model
from keras.preprocessing import sequence
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Dropout, Embedding, Input
from keras.utils.data_utils import get_file
from keras import optimizers, metrics
from sklearn import preprocessing, model_selection

import numpy as np
import random
import sys
import os
import spacy 
import copy


maxlen = 70

ds = Response_Dataset()
list_words = ds.all_words
X = pad_vec_sequences2(ds.all_sentences)
Y = ds.all_next_words
#So every sentence is mapped to the word following it.

x_train, x_test, y_train, y_test = model_selection.train_test_split(X,Y,test_size=0.2)

#build the model: 2 stacked LSTM cells
sequence = Input(shape=(maxlen,300), dtype='float32', name='input')
forwards1 = LSTM(256, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, return_sequences=True, dropout=0.1, recurrent_dropout=0.1)(sequence)
forwards2 = LSTM(256, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, return_sequences=False, dropout=0.1, recurrent_dropout=0.1)(forwards1)
after_dp2 = Dropout(0.5)(forwards2)
output = Dense(300, activation='sigmoid', name='activation')(after_dp2)

model = Model(inputs=sequence, outputs=output)
optimizers.Adam(lr=0.0001, beta_1=0.6, beta_2=0.099, epsilon=1e-08, decay=0.005)
model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

#Comment out the below lines if this is the first time you're training on new data.
#if os.path.isfile('backup/response_models/TextGeneration'):
 #   model.load_model('backup/response_models/TextGeneration.h5')

x_train = np.asarray(x_train)
x_train.ravel()

y_train = np.asarray(y_train)
#y_train.ravel()
# train the model, output generated text after each iteration
no_of_iterations = 10
batch_size = 128
num_epoch = 3
for iteration in range(1, no_of_iterations):
	print()
	print('-' * 50)
	print('Iteration', iteration)
	model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epoch, validation_data=[x_test, y_test])
	model.save('backup/response_models/Textweights.h5',overwrite=True)

	start_index = random.randint(0, len(list_words) - maxlen - 1)

	for diversity in [0.3,0.6, 1.3, 1.6, 1.9]:
		print('Diversity:', diversity)
		generated = ''
		seed = []
		sentence = list_words[start_index: start_index + maxlen]
		for token in sentence:
			seed.append(token.text)
		seed = " ".join(seed)
		#generated += ' '.join(str(word) for word in sentence)
		print('Seed: "' , sentence , '"')
		print("Generated: \n")
		seed_vecs = []
		
		for i in range(200):
			seed_vecs.append(get_sentence_vector(seed))
			x_seed = pad_vec_sequences2(seed_vecs)
			preds = model.predict(x_seed, verbose=0)[0]
			#Preds is in a word vector format.
			#Convert preds into a word here somehow.
			next_word = vec_to_word(preds)
			generated += next_word
			del sentence[0]
			sentence.append(next_word)
			sys.stdout.write(' ')
			sys.stdout.write(next_word)
			sys.stdout.flush()
		print()
