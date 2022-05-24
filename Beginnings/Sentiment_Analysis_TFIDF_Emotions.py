import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.stem import WordNetLemmatizer
from string import punctuation
from autocorrect import Speller
import emoji
import re

from sklearn.model_selection import train_test_split

data = pd.read_csv('text_emotion.csv', encoding = 'ISO-8859-1',
                   names=["tweet_id", "sentiment", "author","content"], skiprows=1)
# we don't want the first row
#print(data.head(10))
data_1 = data.copy()
data_1 = data_1[['sentiment', "content"]]
#print(data_1.head(10))

emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad',
          ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
          ':-@': 'shocked', ':@': 'shocked',':-$': 'confused', ':\\': 'annoyed',
          ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
          '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
          '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink',
          ';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel','(:-D': 'gossip',
          '=^.^=': 'cat', ':D': 'smile', ';D': 'smile'}

def cleaner(tweet):
    """Function to clean text data"""

    for emoj in emojis.keys():
        if emoj in tweet:
            tweet = tweet.replace(emoj, "emoji" + emojis[emoj])

    tweet = tweet.lower()
    tweet = ''.join(c for c in tweet if not c.isdigit())  # remove digits

    tweet = re.sub("@[A-Za-z0-9]+", "", tweet)  # Remove @ sign
    tweet = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", tweet)  # Remove http links

    tweet = " ".join(tweet.split())
    tweet = ''.join(c for c in tweet if c not in emoji.UNICODE_EMOJI)  # Remove Emojis

    tweet = tweet.replace("#", "").replace("_", " ")  # Remove hashtag sign but keep the text
    tweet = ''.join(c for c in tweet if c not in punctuation)  # remove all punctuation

    wordnet_lemmatizer = WordNetLemmatizer()  # with use of morphological analysis of words
    tweet = [wordnet_lemmatizer.lemmatize(word) for word in nltk.word_tokenize(tweet)]

    tweet = " ".join(w for w in tweet)
    return tweet

#data_1_small = data_1.head(90_000)
#data_2 = data_1_small.copy()
data_2 = data_1.copy()
data_2['content'] = data_2['content'].apply(cleaner)

print(data_2[["sentiment","content"]])

from tensorflow.keras.optimizers import SGD



X = data_2['content']
y = data_2['sentiment']
print(y)
print(set(data_2['sentiment']))

emotion_id_mapping = dict()
for i,emotion in enumerate(list(set(data_2['sentiment']))):
    emotion_id_mapping[emotion] = i
print(emotion_id_mapping)

y = [emotion_id_mapping[yi] for yi in y]
#print("new y=",y)
#print(emotion_id_mapping["love"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=65)

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

vectorizer = CountVectorizer()
tfidf = TfidfTransformer()
clf = LogisticRegression(max_iter=10000)

pipeline = Pipeline([
('vec', vectorizer),  # strings to token integer counts
('tfidf', tfidf),  # integer counts to weighted TF-IDF scores
('classifier', clf),  # train on TF-IDF vectors
])

pipeline.fit(X_train, y_train)
pred = pipeline.predict(X_test)
print(classification_report(y_test, pred))
input_user = ''
while (input_user != 'terminate the code') or (input_user != 'terminate'):
    input_user_old = list(str(input('Type your sentence: ')))
    input_user = "".join(input_user_old[0:len(input_user_old)])
    input_user = [input_user]
    print(input_user)
    pred_test = pipeline.predict(input_user)

    print(pred_test)
