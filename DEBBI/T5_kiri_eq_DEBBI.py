# author Eshan K Kaushal

import random
import pandas as pd
import tensorflow as tf
from Sugg_Sys_DEBBI import system_suggs_bro
# multi-head attention required for finding relations
# layer Norm done after every multi-ead attention and ff
# dropout for combating overfitting
# layer - for nn layers
from keras.layers import MultiHeadAttention, LayerNormalization, Dropout, Layer
from keras.layers import Embedding, Input, GlobalAveragePooling1D, Dense
from keras.models import Sequential, Model, load_model
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize

exit_T5 = ['ok then', 'ok', 'okay', 'okay then', 'oh alright', 'alright', 'alright then', 'haha', 'i get it', 'i understand', 'ah thats nice', 'ah thats good'
           'oh i got it', 'nothing', 'ok nice', 'ah nice', 'that is good', 'really nice', 'oh ok', 'oh okay', 'ah okay', 'ah ok', 'thats good', 'got it', 'oh thanks',
           'its fine', 'no problem', 'nice', 'haha nice', 'pretty nice', 'thats so good', 'thats really good', 'aw thats nice', "that's good", "perfect"]
indirect_words = ['they', 'them', 'he', 'she', 'her', 'his', 'their']
exit_T5_abs = ['thanks so much', 'thank you so much', 'thank you', 'see you', 'alright see you then', 'ill see you later then']

from transformers import T5ForConditionalGeneration, T5Tokenizer
model_T5_kiri = T5ForConditionalGeneration.from_pretrained("kiri-ai/t5-base-qa-summary-emotion")
tokenizer_T5_kiri = T5Tokenizer.from_pretrained("kiri-ai/t5-base-qa-summary-emotion")

rec_q = ['', '']
rec_a = ['', '']

def get_answer(question, prev_qa, context):
    input_text = [f"q: {qa[0]} a: {qa[1]}" for qa in prev_qa]
    input_text.append(f"q: {question}")
    input_text.append(f"c: {context}")
    input_text = " ".join(input_text)
    #print('\n input text------------------------------------------------- \n', input_text, '\n\n')
    features = tokenizer_T5_kiri([input_text], return_tensors='pt')
    tokens = model_T5_kiri.generate(input_ids=features['input_ids'],
            attention_mask=features['attention_mask'], max_length=64)
    return tokenizer_T5_kiri.decode(tokens[0], skip_special_tokens=True)

q_list = [i for i in range(0, 90)]

df = pd.read_csv('QA-Pairs.csv')
maxlen = 30
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
main_vocab_word_set = set(main_vocab_word)
# print(main_vocab_word)
dict_ser = {}
for i in range(0, len(main_vocab_word)):
    dict_ser[main_vocab_word[i]] = i

def vectorizer_prepender(inp):
    #print(dict_ser)
    vector = []
    for i in word_tokenize(inp):
        i = i.lower()
        vector.append(main_vocab_word.index(i))
    #vector = np.array(vector)
    #print(vector)
    pp = 0
    while len(vector) != maxlen:
        #vector.insert(0, pp)
        vector.append(0)
    #vector = tf.keras.preprocessing.sequence.pad_sequence(vector, maxlen=25)
    a = np.array(vector)
    a = a.reshape(1, 30)
    #print(a[0])
    return a


class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
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
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
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
    def __init__(self, maxlen=30, vocab_size=1_000, embed_dim=32):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = Embedding(input_dim=vocab_size, output_dim=embed_dim)
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


def reloaded_model():
    maxlen = 30
    vocab_size = 1_000
    embed_dim = 32  # Embedding size for each token
    num_heads = 4  # Number of attention heads
    ff_dim = 32  # Hidden layer size in feed forward network inside transformer

    inputs = Input(shape=(maxlen,))
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.1)(x)
    outputs = Dense(93, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

model1 = reloaded_model()
model1.load_weights('local_trained.h5')

# user's input
count_for_t5 = 0
count_for_cond_break = 0
#recorded outputs
rec_preds = ['bos', 'bos', 'bos']

import json
file = open("tag_response_Annotated updt.json", 'r')
tag_response = json.load(file)
context = 999_999
recent_tags = [99999, 999999]
recent_answers_string = ''
flag_in = 0

prediction = 9999

while True:
    result = ''
    inp1 = str(input('User: '))
    inp1.lower()
    inp1_wt = word_tokenize(inp1.lower())
    inp1_wt_set = set(inp1_wt)
    rec_q.append(inp1) # keep track for T5

    if recent_tags[-1] == recent_tags[-2] == recent_tags[-3]:
        print('Lets talk about something else shall we!')
        rand_i = random.choice(q_list)
        sugg2 = system_suggs_bro(rand_i)
        print(sugg2)

    if inp1_wt_set.issubset(main_vocab_word_set):
        prediction = np.argmax(model1.predict(vectorizer_prepender(inp1.lower())))

        print(f'***** {prediction} *****')

        if prediction == 0:
            flag_in = 1
            context = 0
            recent_tags.append(prediction)
            result = tag_response['tr'][0]["0"][0]["0"]

        elif prediction == 1:
            flag_in = 1
            context = 1
            recent_tags.append(prediction)
            result = tag_response['tr'][0]["1"][0]["0"] + tag_response['tr'][0]["1"][0]["1"] + tag_response['tr'][0]["1"][0]["2"] + \
                     tag_response['tr'][0]["1"][0]["3"] + tag_response['tr'][0]["1"][0]["4"]

        elif prediction == 2:
            flag_in = 1
            context = 2
            recent_tags.append(prediction)
            result = tag_response['tr'][0]["2"][0]["0"]

        elif prediction == 3:
            flag_in = 1
            context = 3
            recent_tags.append(prediction)
            result = tag_response['tr'][0]["3"][0]["0"]

        elif prediction == 4:
            flag_in = 1
            context = 4
            recent_tags.append(prediction)
            result = tag_response['tr'][0]["4"][0]["0"]

        elif prediction == 5:
            flag_in = 1
            context = 5
            recent_tags.append(prediction)
            result = tag_response['tr'][0]["5"][0]["0"]

        elif prediction == 6:
            flag_in = 1
            recent_tags.append(prediction)
            result = 'Thats good! I am good too! Thanks for asking!'

        elif prediction == 7:
            flag_in = 1
            context = 7
            recent_tags.append(prediction)
            #result = tag_response['tr'][0]["7"][0]["0"]
            result = 'It has been good! I like it here. I am pretty busy with my education and that is good!'

        elif prediction == 8:
            flag_in = 1
            context = 8
            recent_tags.append(prediction)
            result = tag_response['tr'][0]["8"][0]["0"] + tag_response['tr'][0]["8"][0]["1"] + \
                     tag_response['tr'][0]["8"][0]["2"] + tag_response['tr'][0]["8"][0]["3"]

        elif prediction == 9:
            flag_in = 1
            context = 9
            recent_tags.append(prediction)
            result = 'I am a student as of now pursuing my degree. I am to be successful in the future.'

        elif prediction == 10:
            flag_in = 1
            context = 10
            recent_tags.append(prediction)
            #result = tag_response['tr'][0]["10"][0]["0"]
            result = 'My dad works a decent job in Afghanistan working with the Office of Culture and Information. He is a nice guy and work along my mother.'

        elif prediction == 11:
            flag_in = 1
            context = 11
            recent_tags.append(prediction)
            result = tag_response['tr'][0]["11"][0]["0"]

        elif prediction == 12:
            flag_in = 1
            context = 12
            recent_tags.append(prediction)
            # result = tag_response['tr'][0]["10"][0]["0"]
            result = 'My dad works as decent job' + tag_response['tr'][0]["11"][0]["0"]

        elif prediction == 13:
            flag_in = 1
            context = 13
            recent_tags.append(prediction)
            # result = tag_response['tr'][0]["10"][0]["0"]
            result = 'I am very positive about the future. I am working hard and I believe in myself and I am sure I will make my parents proud!'

        elif prediction == 14:
            flag_in = 1
            context = 14
            recent_tags.append(prediction)
            result = tag_response['tr'][0]["14"][0]["0"]

        elif prediction == 15:
            flag_in = 1
            context = 15
            recent_tags.append(prediction)
            result = tag_response['tr'][0]["15"][0]["0"]

        elif prediction == 16:

            context = 16
            recent_tags.append(prediction)
            print('It was nice talking to you! Bye!')
            break

        elif prediction == 17:
            flag_in = 1
            context = 17
            recent_tags.append(prediction)
            result = tag_response['tr'][0]["17"][0]["0"]

        elif prediction == 18:
            flag_in = 1
            context = 18
            recent_tags.append(prediction)
            result = tag_response['tr'][0]["18"][0]["0"] + tag_response['tr'][0]["18"][0]["1"]

        elif prediction == 19:
            flag_in = 1
            context = 19
            recent_tags.append(prediction)
            result = tag_response['tr'][0]["17"][0]["1"]

        elif prediction == 20:
            recent_tags.append(prediction)
            result = 'Alright then!'

        elif prediction == 21:
            recent_tags.append(prediction)
            result = 'Thanks! I am happy to hear that.'

        elif prediction == 22:
            recent_tags.append(prediction)
            result = 'Sure!'

        elif prediction == 23:
            flag_in = 1
            context = 23
            recent_tags.append(prediction)
            result = tag_response['tr'][0]["23"][0]["1"]

        elif prediction == 33:
            flag_in = 1
            context = 33
            recent_tags.append(prediction)
            result = 'I think a society where everyone has equal rights and opportunities and where there is no fear of crime and where everyone is accepted irrespective of their background or their status is a Good Society in my opinion.'

        elif prediction == 57:
            flag_in = 1
            context = 57
            recent_tags.append(prediction)
            result = tag_response['tr'][0]["57"][0]["1"] + tag_response['tr'][0]["57"][0]["2"]

        elif prediction == 68:
            context = 68
            flag_in = 1
            recent_tags.append(prediction)
            result = tag_response['tr'][0]["68"][0]["0"] + tag_response['tr'][0]["68"][0]["1"] + tag_response['tr'][0]["68"][0]["2"]

        elif prediction == 71:
            flag_in = 1
            context = 71
            recent_tags.append(prediction)
            result = tag_response['tr'][0]["71"][0]["0"]

        elif prediction == 85:
            flag_in = 1
            recent_tags.append(prediction)
            result = 'I dont understand your question. Please rephrase or ask me something else.'

        elif prediction == 90:
            recent_tags.append(prediction)
            result = "You're welcome!"

        elif prediction == 91:
            flag_in = 1
            recent_tags.append(prediction)
            result = "<Fill in Education Degree name>"


        elif (prediction in [25, 73, 74, 75, 76, 78, 80, 81, 82, 83, 87]):
            if flag_in == 0:
                recent_tags.append(prediction)
                print('Please ask me a question first. I dont have enough background to understand this question.')
            else:
                print('flag_in triggered')
                recent_tags.append(prediction)
                ans_kiri = get_answer(rec_q[-1], [(rec_q[-2], rec_a[-1])], recent_answers_string)
                #result = str(ans_kiri)
                print(ans_kiri)
                flag_in = 0

        elif (prediction == 80 or 'it' in inp1_wt) and flag_in == 1:
            print('flag_in triggered')
            recent_tags.append(prediction)
            ans_kiri = get_answer(rec_q[-1], [(rec_q[-2], rec_a[-1])], recent_answers_string)
            #result = str(ans_kiri)
            if 'unknown' in word_tokenize(ans_kiri):
                print('I dont have any answers to that! But you can certainly ask me something else to keep the conversation going!')
            else:
                print(ans_kiri)
            flag_in = 0

    elif (not (inp1_wt_set.issubset(main_vocab_word_set))):
        from context_fat_rizz import context_fat
        ans_kiri = get_answer(rec_q[-1], [(rec_q[-2], rec_a[-1])], context_fat[context][0])
        #result = str(ans_kiri)
        print(ans_kiri)
        flag_in = 1

    if rec_a[-1] == 'unknown':
        print('I dont have any specific answers to that. Please ask me something else!')

    rec_a.append(result) # keep track for T5
    recent_answers_string = ' '.join(rec_a)
    print('RESULT: ', result, '-', flag_in)