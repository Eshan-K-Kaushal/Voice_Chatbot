import json
import nltk
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from nltk.stem import WordNetLemmatizer
import random
import pickle
#import My_Chatbot_Model_NN


lemmer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_data.h5')

def clean(sentence):
    sent_words = nltk.word_tokenize(sentence)
    sent_words = [lemmer.lemmatize(word) for word in sent_words]
    return sent_words

def bag_of_words(sentence):
    sent_word = clean(sentence)
    bag = [0]*len(words)
    for w in sent_word:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    er_thresh = 0.25 # at 25%
    results = [[i, r] for i , r in enumerate(res) if r > er_thresh]
    # get vals only above the confidence val or error threshold
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent':classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_repsonse(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


message = ''
print('Running....\n')

while (message != 'bye now' and message != 'terminate the code' and message != 'Bye now'):
    #message = input('')
    ints = predict_class(message)
    #print('ints:', ints)
    res = get_repsonse(ints, intents)
    print(res)

