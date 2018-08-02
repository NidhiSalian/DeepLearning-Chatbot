# DeepLearning-Chatbot

A simple, experimental chatbot implementation using Deep Learning, implemented in Python.

Current functionalities include:

    Intent Classification [Using a Bidirectional LSTM based RNN - Keras with Tensorflow backend]
    Entity Recognition [Using the Spacy NLP library for Python]
    Dependency Tree Display [Using the Spacy NLP Library for Python]
    Response Generation [Random response generation using Stacked LSTM based RNN - Keras with Tensorflow backend]
    
## Getting Started:

   You will need a working installation of Python 3.5. (Preferrably on Anaconda - [Installation Guide](https://docs.continuum.io/anaconda/install/))
   
   The other requirements(Python Libraries) can be found [here](./requirements.md).

## What This Repo Contains:

_retrain_ner.py_ - Official sample script to retrain existing Spacy model's NER

_dependency_tree.py_ - Contains functions for parsing and printing dependency trees using Spacy and NLTK

_preprocessor.py_ -  Contains functions for loading datasets for intent classification, padding sentence vectors and creating one hot vectors for class labels

_intent_train.py_ - Trains model to detect intents in classes provided.

_intent_predict.py_ - Predicts intent of test queries, recognizes entities and displays the dependency tree

_corpuscleaner.py_ - When beginning to work with a new corpus, run the text file through corpuscleaner.py. This will remove all 
                     numbers, the words "chapter" and "book", and any additional strings specified by the user in the file.

_response_train.py_ - Trains a model to generate random text in the style of data provided. 

_response_generate.py_ - Generates random response based on user input seed.

_backup/_ - Generated models are saved to their respective folder here.

_data/text_generation/_ - Contains a number of possible corpuses as plain text files.

_data/intent_classes/_ - Contains text files, each with numerous sentences corresponding to a particular class of intent. 

## Usage:

Provide your training datasets in the data/ folder.

Format: 

1. For intent classification:

    Each text file contains one sentence per line that is relevant to one particular class of intent. The textfile must be named after the intent that its sentences represent. The names of these text files will then be considered as the classes of intents, later used for training and prediction. The classification model uses a single bidirectional LSTM cell, followed by a dropout layer, with an Adam optimizer. Spacy's default pretrained vectorizer model for English is currently used for word embedding. Future commits will implement [this](https://github.com/explosion/sense2vec) model for more accurate results. 
    
    The current dataset currently provided in data/intent_classification/ is small, but more data can be added or new training datasets can be used. The preprocessor.py file can be adjusted accordingly to read training input.

2. For text generation:

    Provide as much dialogue data as possible in a text file, as a sample for the type of text you want to generate.
    The current files provided include a variety of training data, each of which will result in a wildly different model, with respect to the style of responses generated.The current implementation of the response generation phase is a very basic word sequence generator, that uses 2 stacked LSTM cells, each followed by a dropout layer.
    
    Much larger training relevant corpuses are required in order to train a robust, domain-specific chatbot. Best trained on a GPU, because this process is computation intensive.(Note: If you are using a GPU, use the Theano backend with Keras by changing the 'backend' attribute in your .keras/keras.json file to theano. You may also need to install the package mkl-service on Linux systems. )
    
    The current implementation is in the form of a word - level RNN. A [sequence level model](http://suriyadeepan.github.io/2016-06-28-easy-seq2seq/) would probably be more appropriate and will be incorporated in future commits.
    
    Further commits will incorporate the response selector pipeline, that will ensure that the responses are more relevant to the user's query.
    
##### To train the intent classification model on the training data:

python intent_train.py

##### To test the intent classification model:

python intent_predict.py

##### To train the response generation model:

python response_train.py

##### To generate a _random_ response from the trained model:

python response_generate.py
    
## Built With:

   [Anaconda](https://docs.continuum.io/anaconda/install/)     [Keras](https://github.com/fchollet/keras/tree/master/docs)     [Spacy](https://spacy.io/)
   
## Acknowledgments:
   _response_train.py_ 
    Based on [this Keras example](https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py)
    
   _corpuscleaner.py_ 
    From Maia McCormick's [project](https://github.com/maiamcc/markovgen)
    
   _retrain_ner.py_
    Official sample script from [Spacy examples](https://github.com/explosion/spacy/blob/master/examples/training/train_new_entity_type.py)

## Note:
 The code for this project has not been updated since September 2017 . 
 If you have any queries on this project, you can contact me at: nidhisalian08@gmail.com .

## License:

[MIT License](./LICENSE)
