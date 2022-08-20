# author: Eshan dated: 08/15/2022

# dependencies
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
# importing all the libraries
from context_test_wiki import context
import json
import numpy as np
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
import random
import pickle
import docx2txt
import re
from nltk.tokenize import sent_tokenize
import random
import cv2
from ffpyplayer.player import MediaPlayer

from transformers import pipeline
nlp = pipeline("question-answering")

res_recent = [] # list to keep track of the recent tags

# ----------function to play videos----------
def PlayVideo(video_path):
    video=cv2.VideoCapture(video_path)
    player = MediaPlayer(video_path)
    while True:
        grabbed, frame=video.read()
        audio_frame, val = player.get_frame()
        if not grabbed:
            print("End of video")
            break
        if cv2.waitKey(28) & 0xFF == ord("q"):
            break
        cv2.imshow("Video", frame)
        if val != 'eof' and audio_frame is not None:
            #audio
            img, t = audio_frame
    video.release()
    cv2.destroyAllWindows()

# --------------- to reply with something as follows if they talk about something more than 3 times ---------------
def Talk_Something_Else():
    talked_a_lot = ['We have talked about this a lot, lets talk about something else!',
                    'I say, lets move on to a new topic.'
                    'Okay, lets talk about something else', 'Lets move on to a new topic']
    return random.choice(talked_a_lot)

# ---------------- for replying with something if the user says something positive ---------------
def Gratitude():
    gc = ['Yes!', 'Thanks!', 'I know right!']
    return random.choice(gc)

# ------ loading the model ------------
model = load_model('/content/Voice_Chatbot/Cleaned_Code/trained_questions.h5')

intents_out = json.loads(open('/content/Voice_Chatbot/Cleaned_Code/questions.json').read())

# -------class of all functions that serve as helper functions to prepare the input sentence for running the predictor on it--------
class Model_Helpers:
    # GLOBAL
    lemmer = WordNetLemmatizer()
    intents = json.loads(open('/content/Voice_Chatbot/Cleaned_Code/questions.json').read())

    words = pickle.load(open('/content/Voice_Chatbot/Cleaned_Code/words.pkl', 'rb'))
    classes = pickle.load(open('/content/Voice_Chatbot/Cleaned_Code/classes.pkl', 'rb'))

    def clean(self, sentence):
        sent_words = nltk.word_tokenize(sentence)
        sent_words = [self.lemmer.lemmatize(word) for word in sent_words]
        return sent_words

    def bag_of_words(self, sentence):
        sent_word = self.clean(sentence)
        bag = [0] * len(self.words)
        for w in sent_word:
            for i, word in enumerate(self.words):
                if word == w:
                    bag[i] = 1
        return np.array(bag)

    def predict_class(self, sentence):
        bow = self.bag_of_words(sentence)
        res = model.predict(np.array([bow]))[0]
        er_thresh = 0.45  # at 25%
        results = [[i, r] for i, r in enumerate(res) if r > er_thresh]
        # get vals only above the confidence val or error threshold
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({'intent': self.classes[r[0]], 'probability': str(r[1])})
        return return_list

    def get_repsonse(intents_list, intents_json):
        tag = intents_list[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                break
        return result
