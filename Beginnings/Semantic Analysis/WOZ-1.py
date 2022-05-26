# 0 - intro
# 1 - Reply to bot's question
# 2 - Asking the bot how's it doing
# 3 - Experience - in the new country
# 4 - Experience - in the old country
# 5 - Dad's job
# 6 - How do you support yourself
# 7 - Do you like your work
# 8 - Where are your right now - location
# 9 - What your future looks like
# 10- Talk of Kids
# 11- Favorite activity

import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from string import punctuation
import emoji
import re

from sklearn.model_selection import train_test_split

data = pd.read_csv('intents.csv', encoding = 'ISO-8859-1',
                   names=["sentiment","content"], skiprows=1)
# we don't want the first row
#print(data.head(10))
data_1 = data.copy()
data_1 = data_1[['sentiment', "content"]]
#print(data_1.head())

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

# print(data_2[["sentiment","content"]])
# print(data_2.head())

data_2['content'] = data_2['content'].apply(cleaner)

# print(data_2[["sentiment","content"]])

X = data_2['content']
y = data_2['sentiment']

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
    input_user_old = list(str(input('Type your sentence (USER): ')))
    input_user = "".join(input_user_old[0:len(input_user_old)])
    input_user2 = input_user
    print(input_user2)
    input_user = [input_user]
    #print(input_user)
    pred_test = pipeline.predict(input_user)

    if pred_test == [0]:
        print('Bot: Hi! How are you?')
    elif pred_test == [1]:
        print("That's great!")
    elif pred_test == [2]:
        print('I am doing good! Thanks for asking!')
    elif pred_test == [3]:
        print('It has been good so far. I am really happy here and pretty satisfied with my given condition!')

    elif pred_test == [4]:
        print('It was not that good back there. A sense of fear always accompanied me anywhere I went.')
    elif pred_test == [5]:
        print('My Dad worked as an Official in the Foreign Administrative Services')
    elif pred_test == [6]:
        print('I work at CVS')
    elif pred_test == [7]:
        print('Its okay, I would like to work as a mechanical engineer to be honest')
    elif pred_test == [8]:
        print('I am currently talking to you in this here museum')
    elif pred_test == [9]:
        print('I am pretty positive about it')
    elif pred_test == [10]:
        print('I have 4 kids: 2 sons and 2 daughters')
    elif pred_test == [11]:
        print('I love to race cars on the weekends. Being a mechanical engineer for Ford allows me to test'
              'cars on a regular basis')
