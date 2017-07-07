from preprocessor import pad_vec_sequences, labels
from keras.models import load_model
import spacy
from preprocessor import nlp
import numpy as np
from dependency_tree import to_nltk_tree , to_spacy_desc

nb_classes = len(labels)

#load the model to be tested.
model = load_model('backup/intent_models/model2.h5')

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
	
#convert all the sentences into matrices of equal size.
test_vec_seq = pad_vec_sequences(test_vec_seq)

#get predictions
prediction = model.predict(test_vec_seq)

label_predictions = np.zeros(prediction.shape)

for i in range(n_test):
	m = max(prediction[i])
	p = np.where(prediction[i] > 0.55 * m)	# p collects possible sub intents
	q = np.where(prediction[i] == m)	#q collects intent
	label_predictions[i][p] = 1
	label_predictions[i][q] = 2

'''
 label_predictions[i] is now a list of intent predictions for Query i, where:
 One of the following integer values is saved per intent label:
 2 - This is the most probable intent of the given query
 1 - This could possibly be a sub intent of the given query
 0 - This intent is not present in the given query 
'''
#print("Classes: ", nb_classes)
#print("Labels", len(label_predictions[i]))

for i in range(n_test):
	print("\n Displaying Predictions:")
	print(" Query ", i+1, " :", test_seq[i])
	print(" Entities Recognized:", end = "\t")
	if len(test_ent_seq[i]) == 0:
		print(" None.", end = "\t")
	for ent in test_ent_seq[i]:
		print(" ",ent.label_, ent.text, end= "\t")
	print("\n Dependency tree:")
	tx = nlp(test_seq[i])
	[to_nltk_tree(sent.root).pretty_print() for sent in tx.sents]
	#to_spacy_desc(tx)	#Prints subject - activity - numbers
	#Using the to spacy desc function is more useful in general statements. 
	#I will include this as a functionality once there is enough data to incorporate such a class.
	for x in range(len(label_predictions[i])):
		if label_predictions[i][x] == 2 :
			print(" Detected intent: ",labels[x])
		if label_predictions[i][x] == 1:
			print(" Detected possible sub-intent: ",labels[x])
	if  len(set(label_predictions[i])) == 1:
		print(" Could not detect intent. ")
		
print("\nTest Complete")
