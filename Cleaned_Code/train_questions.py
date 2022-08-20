# author: Eshan dated: 08/15/2022

# the neural network is going to use the intents.json file
# it will find all the statement patterns under a tag
# it will then tokenize those statements and make groups of them
# it will then train the neural network on these groups of words for the given tag
# once all the tags has been trained in a similar way, it will be fully trained
# with the trained model file in hand and loaded, it will then look for the inputs by the user
# the neural will then analyze the sentence (which will be in the tokenized form) and would try
# to classify the given group of words as the input into one of the categories/tags.
# The neural network being a classifier by nature, would then classify the given input sentence
# into one of the tags and once the tag has been decided for the input sentence, it then
# would give out the response wrt that tag that was predicted for the input sentence.

import random
import re
import tensorflow as tf
import keras.optimizers
import nltk
import numpy as np
import json
from nltk.stem import WordNetLemmatizer
from tensorflow import keras
import pickle
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
from keras.optimizers import Adam
#--------------------------------------------------------------------------------------------------
lemmer = WordNetLemmatizer()
fintents = json.loads(open('questions.json').read())

words = []
classes = []
documents = []
ignore = ['?', '.', '!', ',', "'"]

for intent in fintents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list) # in order to add more to the list we want to take a thing and "add"
        # it to the list
        documents.append((word_list, intent['tag'])) # breaking all the patterns into words and then setting
        #them up class/tag wise
        if intent['tag'] not in classes:
            classes.append(intent['tag']) # we make a list of tags too, to keep track of all the classes
            #we have encountered so far

words = [lemmer.lemmatize(word) for word in words if word not in ignore]
words = sorted(set(words))
classes = sorted(set(classes)) # making a sorted list of classes - 'set' removes all the repeating

pickle.dump(words, open('words.pkl', 'wb')) #store them in a file
pickle.dump(classes, open('classes.pkl', 'wb')) #store them in a file

training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    word_patterns = doc[0]
    word_patterns = [lemmer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1 #set 1 for true label
    training.append(([bag, output_row]))
random.shuffle(training)
training = np.array(training)

train_x = list(training[:,0])
train_y = list(training[:,1])

# neural net

model = Sequential()
model.add(Dense(256, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax')) # we use softmax for categorical data
# size of the last layer is equal to the number of labels/category of the data
sgd = SGD(lr = 0.0001, decay=1e-6, momentum=0.9, nesterov=True) # momentum and nestrov are internet std vals
model.compile(loss = 'categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
mod = model.fit(np.array(train_x), np.array(train_y), epochs=8000, batch_size=3, verbose= 1)
model.save('trained_questions.h5', mod)
print('Fin')