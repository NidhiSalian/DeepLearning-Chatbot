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

_preprocessor.py_  Loads datasets for intent classification, pads sentence vectors and creates one hot vectors for class labels

_intent_train.py_ Trains model to detect intents in classes provided.

_intent_predict.py_ Predicts intent of test queries, recognizes entities and displays the dependency tree

_response_train.py_ Trains a model to generate random text in the style of data provided.

_response_generate.py_ Generates random response based on user input seed.

## Usage:

Provide your training datasets in the data/ folder.

Format: 

1. For intent classification:

    Each text file contains one sentence per line that is relevant to one particular class of intent. The textfile must be named after the intent that its sentences represent. The names of these text files will then be considered as the classes of intents, later used for training and prediction. The classification model uses a single bidirectional LSTM cell, followed by a dropout layer, with an Adam optimizer.
    
    The current dataset currently provided in data/intent_classification/ is small, but more data can be added or new training datasets can be used. The preprocessor.py file can be adjusted accordingly to read training input.

2. For text generation:

    Provide as much dialogue data as possible in a text file, as a sample for the type of text you want to generate.
    The current files provided include a variety of training data, each of which will result in a wildly different model, with respect to the style of responses generated.
    
    Much larger training relevant corpuses are required in order to train a robust, domain-specific chatbot. The current implementation of the response generation phase is a very basic word sequence generator, that uses 2 stacked LSTM cells, each followed by a dropout layer.
    
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

   [Anaconda](https://docs.continuum.io/anaconda/install/)
   
   [Keras](https://github.com/fchollet/keras/tree/master/docs)
   
   [Spacy](https://spacy.io/)
   
## License:

[MIT License](./LICENSE)
