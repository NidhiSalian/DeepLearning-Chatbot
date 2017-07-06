import spacy
from os import listdir
from os.path import isfile, join
import numpy as np

nlp = spacy.load('en')
print("Loaded Vectorizer.")

data_path = 'data/intent_classes/'
labels = [f.split('.')[0] for f in listdir(data_path) if isfile(join(data_path, f))]

class Dataset(object):
	def __init__(self):
		vocab = nlp.vocab
		X_all_sent = []
		X_all_vec_seq = []
		X_all_doc_vec = []
		Y_all = []
		for label in labels:
			x_file = open('data/'+label + '.txt') 
			x_sents = x_file.read().split('\n')
			for x_sent in x_sents:
				if len(x_sent) > 0:
					x_doc = nlp(x_sent)
					x_doc_vec = x_doc.vector	
					x_vec_seq = []
					for word in x_doc:
						
						x_vec_seq.append(word.vector)
						print(x_vec_seq) 
						X_all_sent.append(x_sent)
						X_all_doc_vec.append(x_doc_vec)
						X_all_vec_seq.append(x_vec_seq)
						Y_all.append(label)

		self.X_all_sent = X_all_sent
		self.X_all_vec_seq = X_all_vec_seq
		self.X_all_doc_vec = X_all_doc_vec
		self.Y_all = Y_all

def pad_vec_sequences(sequences,maxlen=50):
	new_sequences = []
	for sequence in sequences:
		
		orig_len, vec_len = np.shape(sequence)
		if orig_len < maxlen:
			new = np.zeros((maxlen,vec_len))
			new[maxlen-orig_len:,:] = sequence
		else:
			#print(sequence)
			new = sequence[orig_len-maxlen:,:]
		new_sequences.append(new)
	new_sequences = np.array(new_sequences)
	#print(new_sequences.shape)
	return new_sequences
	
def pad_class_sequence(sequence, nb_classes):
	return_sequence = []
	for label in sequence:
		new_seq = [0.0] * nb_classes
		if label in labels:
			new_seq[labels.index(label)] = 1.0
		return_sequence.append(new_seq)
	return return_sequence
		
