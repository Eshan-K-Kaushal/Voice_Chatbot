# author Eshan K Kaushal

import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer, BertModel
import torch
import nlpaug.augmenter.word as naw
# multi-head attention required for finding relations
# layer Norm done after every multi-ead attention and ff
# dropout for combating overfitting
# layer - for nn layers
from keras.layers import MultiHeadAttention, LayerNormalization, Dropout, Layer
from keras.layers import Embedding, Input, GlobalAveragePooling1D, Dense
from keras.datasets import imdb
from keras.models import Sequential, Model, load_model
from sklearn.model_selection import train_test_split
import numpy as np
import warnings
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

df = pd.read_csv('QA-Pairs.csv')

class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.01):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential( # the feed forward neural network
            [Dense(ff_dim, activation="relu"),
             Dense(ff_dim, activation="relu"),
             Dense(embed_dim),]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output2 = self.att(inputs, inputs)

        #hadamard = tf.multiply(attn_output, attn_output2)

        #addition = torch.add(attn_output, attn_output2)
        #mean_att = torch.div(addition, 2)

        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output) # residual connection
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            #'vocab_size': self.vocab_size,
            #'max_lem': self.max_len,
            #'embed_dim': self.embed_dim,
            #'num_heads': self.num_heads,
            #'ff_dim': self.ff_dim,
            #'rate': self.rate,
            #'num_heads': self.num_heads,
            #'dropout': self.dropout,
        })
        return config

class TokenAndPositionEmbedding(Layer):
    def __init__(self, maxlen=25, vocab_size=1_000, embed_dim=32):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = Embedding(input_dim=vocab_size, output_dim=embed_dim) # working with fixed embeddings
        self.pos_emb = Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

    def get_config(self):
        config = super().get_config()
        config['token_emb'] = self.token_emb
        config['pos_emb'] = self.pos_emb
        #config['maxlen'] = self.maxlen
        #config['vocab_size'] = self.vocab_size
        #config['embed_dim'] = self.embed_dim
        config.update({
            'token_emb': self.token_emb,
            'pos_emb': self.pos_emb,
            #'embed_dim': self.embed_dim,
            #'num_heads': self.num_heads,
            #'units': self.units,
            #'d_model': self.d_model,
            #'num_heads': self.num_heads,
            #'dropout': self.dropout,
        })
        return config

vocab_size = 1_000  # Only consider the top 500 words
maxlen = 30 # Only consider the first 30 words of each input
aug = naw.SynonymAug(aug_src='wordnet', aug_max=2)
X = df['Questions']
y = df['Tags']

print(X.shape)
print(y.shape)

xtrain, xtest, ytrain, ytest = train_test_split(df['Questions'], df['Tags'],
                                                 test_size=0.27)

print(len(xtrain), "Training sequences")
print(len(xtest), "Validation sequences")

# main vocab here-----------------------------------------
main_vocab = []
main_vocab_word = []

for i in df['Questions']:
    i = i.lower()
    main_vocab.append(i)
for i in main_vocab:
    for j in word_tokenize(i):
        main_vocab_word.append(j)

main_vocab_word = list(set(main_vocab_word))
main_vocab_word = sorted(main_vocab_word)
print('Main Vocab:\n', main_vocab_word, 'len: ', len(main_vocab_word))
# main vocab here-----------------------------------------

def vectorizer(inp): # here i am tryin to implement my own custom word embedding
    main_vocab = []
    main_vocab_word = []

    for i in df['Questions']:
        i = i.lower()
        main_vocab.append(i)
    for i in main_vocab:
        for j in word_tokenize(i):
            main_vocab_word.append(j)

    main_vocab_word = list(set(main_vocab_word))
    main_vocab_word = sorted(main_vocab_word)


    dict_ser = {}
    for i in range(0, len(main_vocab_word)):
        dict_ser[main_vocab_word[i]] = i
    #print(dict_ser)
    vector = []
    for i in word_tokenize(inp):
        i = i.lower()
        vector.append(main_vocab_word.index(i))
    #vector = np.array(vector)
    return vector

x_train = []
x_test = []

for i in xtrain:
    x_train.append(vectorizer(i))

for i in xtest:
    x_test.append(vectorizer(i))

print(len(x_train), len(x_test))
print('Embs are: ', x_train)

y_train = np.array(ytrain)
y_test = np.array(ytest)

x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen, padding='post')
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen, padding='post')
print(x_train[0], x_train[1], x_train[2])

embed_dim = 32  # Embedding size for each token
num_heads = 4  # Number of attention heads
ff_dim = 32  # Hidden layer size in feed forward network inside transformer

inputs = Input(shape=(maxlen,))
embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
x = embedding_layer(inputs)

transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(x)
x = GlobalAveragePooling1D()(x)
x = Dropout(0.3)(x)
x = Dense(32, activation="relu")(x)
# # added new below 2 rate 0.01
# x = Dropout(0.1)(x)
# x = Dense(32, activation="relu")(x)
# -----------------------------
x = Dropout(0.1)(x)

outputs = Dense(len(list(set(y_train))), activation="softmax")(x)

model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

history = model.fit(x_train, y_train, batch_size=20, epochs=150, validation_data=(x_test, y_test) )

model.save_weights("local_trained.h5")
