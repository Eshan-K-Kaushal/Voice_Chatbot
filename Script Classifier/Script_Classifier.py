import json
import nltk
import pandas as pd
import numpy as np
from keras.models import load_model
#from tensorflow.keras.models import load_model
from nltk.stem import WordNetLemmatizer
import random
import pickle
import docx2txt
import re
from nltk.tokenize import sent_tokenize
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

text_doc = docx2txt.process('script_test.docx')
#print(text_doc)
text_doc_replace = re.sub('\n', " ", text_doc)
#print(text_doc_replace)
sent = sent_tokenize(text_doc_replace)
#print(sent)
print("--------------------------------")
for i in sent:
    print(i)
    #message = input('')
    ints = predict_class(i)
    #print('ints:', ints)
    res = get_repsonse(ints, intents)
    print("Tag would be:", res)
    print('------------------------------')

