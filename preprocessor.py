import spacy
from os import listdir
from os.path import isfile, join
import numpy as np

nlp = spacy.load('en')
print("Loaded Vectorizer.")
vocab = nlp.vocab

response_data_path = "data/text_generation/new" #path to training file, preferably saved as a .txt file or simply a document.

intent_data_path = 'data/intent_classes/'	#path to intent training files
labels = [f.split('.')[0] for f in listdir(intent_data_path) if isfile(join(intent_data_path, f))]

class Dataset(object):
	def __init__(self):
		X_all_sent = []
		X_all_vec_seq = []
		X_all_doc_vec = []
		Y_all = []
		for label in labels:
			x_file = open(intent_data_path+label + '.txt') 
			x_sents = x_file.read().split('\n')
			for x_sent in x_sents:
				if len(x_sent) > 0:
					x_doc = nlp(x_sent)
					x_doc_vec = x_doc.vector	
					x_vec_seq = []
					for word in x_doc:
						x_vec_seq.append(word.vector)
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
			new = sequence[orig_len-maxlen:,:]
		new_sequences.append(new)
	new_sequences = np.array(new_sequences)
	return new_sequences

def pad_vec_sequences2(sequences,maxlen=70):
	new_sequences = []
	for sequence in sequences:
		
		orig_len, vec_len = np.shape(sequence)
		sequence = np.array(sequence)
		if orig_len < maxlen:
			new = np.zeros((maxlen,vec_len))
			new[maxlen-orig_len:,:] = sequence
		else:
			new = sequence[orig_len-maxlen:,:]
		new_sequences.append(new)
	new_sequences = np.array(new_sequences)
	return new_sequences
	
def pad_class_sequence(sequence, nb_classes):
	return_sequence = []
	for label in sequence:
		new_seq = [0.0] * nb_classes
		new_seq[labels.index(label)] = 1.0
		return_sequence.append(new_seq)
	return return_sequence
	
class Response_Dataset(object):
	
	def __init__(self):
		
		try: 
			text = open(response_data_path).read()
		except UnicodeDecodeError:
			import codecs
			text = codecs.open(response_data_path, encoding='utf-8').read()
    	
		print('corpus length:', len(text))
    	
		text_sents = text.split('\n')
    	
		all_words = []
		all_sentences = []	#stores list of sentences as a sequence of word vectors
		all_next_words = []	#stores list of all first words following a sentence(as a vector)
		first_sentence = 1	#flag. Set to 0 from after processing of first sentence
		for sentence in text_sents:
			if len(sentence) > 0:
				x_sent_nlp = nlp(sentence)
				words_in_sentence = []
						
				for word in x_sent_nlp:
					if word.is_alpha:
						all_words.append(word)
						words_in_sentence.append(word.vector)
				if len(words_in_sentence) > 0:
					all_sentences.append(words_in_sentence)
					if(first_sentence == 0):
						all_next_words.append(words_in_sentence[0])
					else:
						first_sentence = 0
		
		print("Total words:", len(all_words))
		print("Total sentences:", len(all_sentences))
		
		all_sentences.pop() #removes the last sentences, because we would not have a next_word for it.
		
		self.all_words = all_words
		self.all_sentences = all_sentences
		self.all_next_words = np.asarray(all_next_words)
    	
def get_sentence_vector(seed):
	seed_nlp = nlp(seed)
	seed_vec = []
	for word in seed_nlp:
		seed_vec.append(word.vector)
	return np.array(seed_vec)

def vec_to_word(next_word_vec):
	#Given a vector prediction for the next word, match it to a word in the vocab.
	try:
		next_word = vocab.__getitem__(next_word_vec)
	except:
		next_word = "None"
	return next_word
		
