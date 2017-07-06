from preprocessor import Dataset, pad_vec_sequences, labels, pad_class_sequence
from sklearn import preprocessing, model_selection
import numpy as np

from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import Dense, Dropout, Embedding, Input, merge
from keras.layers.recurrent import LSTM, GRU
from keras.layers import Concatenate
from keras.utils import np_utils, generic_utils
from keras import optimizers, metrics



maxlen = 50 #sentences with length > maxlen will be ignored
hidden_dim = 32
nb_classes = len(labels)

ds = Dataset()
print("Datasets loaded.")
X_all = pad_vec_sequences(ds.X_all_vec_seq)
Y_all = ds.Y_all
#print (X_all.shape)
x_train, x_test, y_train, y_test = model_selection.train_test_split(X_all,Y_all,test_size=0.2)
y_train = pad_class_sequence(y_train, nb_classes)
y_test = pad_class_sequence(y_test, nb_classes)
 
#THE MODEL
sequence = Input(shape=(maxlen,300), dtype='float64', name='input')
#forwards lstm
forwards = LSTM(hidden_dim, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, dropout=0.1, recurrent_dropout=0.1)(sequence)
#backwards lstm
backwards = LSTM(hidden_dim, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, dropout=0.1, recurrent_dropout=0.1, go_backwards=True)(sequence)
#forwards + backwards = bidirectional lstm
merged = merge([forwards, backwards])
#dropout layer applied to prevent overfitting
after_dp = Dropout(0.6)(merged)
#softmax activation layer
output = Dense(nb_classes, activation='sigmoid', name='activation')(after_dp)
#Putting it all together

model = Model(inputs=sequence, outputs=output)
#Your model is now a bidirectional LSTM cell + a dropout layer + an activation layer.
optimizers.Adam(lr=0.001, beta_1=0.6, beta_2=0.099, epsilon=1e-08, decay=0.005, clipnorm = 1.)
model.compile(optimizer = 'Adam', loss = 'categorical_crossentropy',metrics=['categorical_accuracy'])

#This is directly from the tutorial:
#Here we chose categorical cross entropy for the loss function and the adam optimizer which is a slightly enhanced version of the stochastic gradient descent.
#Next we trained this classifier over some epochs.

#initialise batch_size
batch_size = 20
#initialise num_epoch
num_epoch = 3

x_train = np.asarray(x_train)
x_train.ravel()

y_train = np.asarray(y_train)
y_train.ravel()

print("Fitting to model")
my_model = model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epoch, validation_data=[x_test, y_test])

print("Model Training complete.")

#save the model
model.save("backup/intent_models/model1.h5")

print("Model saved to backup folder.")

#how to load this model elsewhere
#from keras.models import load_model
#model = load_model('my_model.h5')

#del model  # deletes the existing model
		
		
