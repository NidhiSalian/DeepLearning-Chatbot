from preprocessor import pad_vec_sequences, labels
from keras.models import load_model
import spacy
from preprocessor import nlp
import numpy as np
from dependency_tree import to_nltk_tree #, to_spacy_desc

nb_classes = len(labels)

#load the model to be tested.
model = load_model('backup/intent_models/model1.h5')


n_test = int(input("\nNumber of test queries? \n"))
test_vec_seq = [] #list of all vectorized test queries
test_ent_seq = [] #list of lists of entities in each test query
test_seq = [] #list of all test queries
for i in range(n_test):
	print ("\nEnter Query",i+1," to be classified:")
	test_text = input()
	test_seq.append(test_text)
	#vectorize text.
	test_doc = nlp(test_text)
	test_vec = []
	for word in test_doc:
		test_vec.append(word.vector)
	test_vec_seq.append(test_vec)
	test_ent_seq.append(test_doc.ents)
	
#print (labels)
test_vec_seq = pad_vec_sequences(test_vec_seq)
#print (test_vec_seq)
prediction = model.predict(test_vec_seq)
#print (prediction)
label_predictions = np.zeros(prediction.shape)
#label_predictions[prediction>=0.5] = 1
for i in range(n_test):
	m = max(prediction[i]) #change this to a suitable limit like say, 0.5 once you have enough data to train your classifier properly on multiple intents
	p = np.where(prediction[i] > 0.15 * m)#prediction[i].index(m)
	q = np.where(prediction[i] == m)
	label_predictions[i][p] = 1
	label_predictions[i][q] = 2
#print (label_predictions)
#print (label_predictions)

for i in range(n_test):
	print("\n Displaying Predictions:")
	print(" Query ", i+1, " :", test_seq[i])
	print(" Entities Recognized:")
	if len(test_ent_seq[i]) == 0:
		print(" None.", end = "\t")
	for ent in test_ent_seq[i]:
		print(" ",ent.label_, ent.text, end= "\t")
	print("\n Dependency tree:")
	tx = nlp(test_seq[i])
	[to_nltk_tree(sent.root).pretty_print() for sent in tx.sents]
	#to_spacy_desc(tx)
	for x in range(nb_classes):
		if label_predictions[i][x] == 2 :
			print(" Detected intent: ",labels[x])
		if label_predictions[i][x] == 1:
			print(" Detected sub-intent: ",labels[x])
	if  len(set(label_predictions[i])) == 1:
		print(" Could not detect intent. ")
print("\nTest Complete")
