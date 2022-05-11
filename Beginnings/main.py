# author - Eshan
# 10/05/2022

import speech_recognition as sr
import pyttsx3
import webbrowser as web
import nltk
from nltk.tokenize import word_tokenize
import json
import nltk
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from nltk.stem import WordNetLemmatizer
import random
import pickle
rec = sr.Recognizer()

def SpeakText(command):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('rate', 175)
    engine.setProperty('voice', voices[1].id)
    engine.say(command)
    engine.runAndWait()

# test
# SpeakText('Hello man! Whats up')
# taking the input from the microphone as the source
while(True):
    with sr.Microphone() as source:
        print('Make sure the surroundings are silent...\n')
        rec.adjust_for_ambient_noise(source, duration=3)
        print('Calibrated the microphone - you can speak now...\n')

        audio = rec.listen(source)
        # use google to recognize audio
        text_conv = rec.recognize_google(audio)
        text_conv = text_conv.lower()
        print('You said: ',text_conv)
        text_conv_wt = word_tokenize(text_conv)
        # SpeakText(text_conv)
        # opening up websites
        if text_conv == 'youtube' or text_conv == 'open up youtube' or text_conv == 'open youtube' or text_conv == 'give me youtube' \
                or text_conv == 'open up youtube for me' or text_conv == 'open youtube for me':
            SpeakText("Here is the site you requested")
            web.open('https://www.youtube.com/')

        elif text_conv == 'google' or text_conv == 'open up google' or text_conv == 'open google' or text_conv == 'give me google' \
                or text_conv == 'open up google for me' or text_conv == 'open google for me':
            SpeakText("Here is the site you requested")
            web.open('https://www.google.com/')

        elif text_conv == 'covid-19 updates' or text_conv == 'open up covid updates' or text_conv == 'open covid' or text_conv == 'give me updates on covid' \
                or text_conv == 'coronavirus updates' or text_conv == 'get updates on covid for me' or text_conv == 'open up covid-19 updates':
            SpeakText("Here is the information you requested")
            web.open('https://www.cdc.gov/coronavirus/2019-ncov/index.html')

        elif text_conv == 'i need help with my depression' or text_conv == 'i need help with my sadness' or text_conv == 'please help me with my sadness'\
                or text_conv == 'please help me with my depression' or text_conv == 'i am really sad' or text_conv == "i'm really sad":
            # import My_Chatbot_Model_NN

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
                bag = [0] * len(words)
                for w in sent_word:
                    for i, word in enumerate(words):
                        if word == w:
                            bag[i] = 1
                return np.array(bag)


            def predict_class(sentence):
                bow = bag_of_words(sentence)
                res = model.predict(np.array([bow]))[0]
                er_thresh = 0.25  # at 25%
                results = [[i, r] for i, r in enumerate(res) if r > er_thresh]
                # get vals only above the confidence val or error threshold
                results.sort(key=lambda x: x[1], reverse=True)
                return_list = []
                for r in results:
                    return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
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
            print('Hi! I am Lizzy, a multimodal chatbot at your service!\n')
            SpeakText('Hi! I am Lizzy, a multimodal chatbot at your service!')
            print('Hi! What can I help you with today!')
            SpeakText('What can I help you with today!')
            while (message != 'bye now' and message != 'terminate the code' and message != 'Bye now'):
                #rec.adjust_for_ambient_noise(source, duration=2)
                #audio1 = rec.listen(source)
                #text_conv1 = rec.recognize_google(audio1)
                #text_conv1 = text_conv1.lower()
                #message = text_conv1
                #print('You said: '+text_conv1)
                message = input('User: ')
                ints = predict_class(message)
                res = get_repsonse(ints, intents)
                print(res)
                SpeakText(res)


        elif 'bye' in text_conv_wt or 'terminate' in text_conv_wt:
            print('Bye Now!')
            SpeakText('Bye Now!')
            break


