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
import numpy as np
import json
from nltk.stem import WordNetLemmatizer
#from torch import nn
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from nltk import word_tokenize
import nltk
from torch.utils.data import DataLoader
#--------------------------------------------------------------------------------------------------
lemmer = WordNetLemmatizer()
fintents = json.loads(open('intents.json').read())

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
    output_row[classes.index(doc[1])] = 1 # set 1 for true label
    training.append(([bag, output_row]))
random.shuffle(training)
print(training[:2])
TRAINING_X = []
TRAINING_Y = []
for i in training:
    TRAINING_X.append(i[0])
for i in training:
    TRAINING_Y.append(i[1])
#TRAINING = TRAINING_X + TRAINING_Y
#print(TRAINING)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#training_tensor = torch.as_tensor(training)
#training = np.array(training, dtype=object)
#train_x = training[:,0]
#train_y = training[:,1]

# print('------------------')
# print('x\n-------------------',TRAINING_X[:2])
# print('y\n----------------',TRAINING_Y[:2])
# print('-------------------')
print(len(TRAINING_X[0]))
print(len(TRAINING_Y[1]))
#trainX = np.array(training[:,0])
#trainY = np.array(training[:,1])


learning_rate = 0.001


class Feedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        # self.fc2 = torch.nn.Linear(self.hidden_size, 1)
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        output = self.softmax(output)
        return output

model = Feedforward(len(TRAINING_X[0]), len(TRAINING_Y[1])).to(device)
# criterion = torch.nn.CrossEntropyLoss()
#criterion(model, training(TRAINING_Y))
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

train_x = torch.FloatTensor(TRAINING_X).to(device)
train_y = torch.FloatTensor(TRAINING_Y).to(device)
tensor_dataset = torch.utils.data.TensorDataset(train_x,train_y)

train_loader = torch.utils.data.DataLoader(tensor_dataset, batch_size=7,
                                           shuffle=True)

#print('here is the trainloader',train_loader)

#print(train_x)
#print(len(train_x))
#print(train_y)

model.eval()
y_pred = model(train_x)
print('here test:', y_pred)
print(y_pred.shape)
print(y_pred)
#before_train = criterion(y_pred.squeeze(), train_y)
before_train = criterion(y_pred, train_y)
print('Test loss before training' , before_train.item())
train_acc = 0
model.to(device)
epoch = 1
for epoch in range(epoch):
    model.train()
    for (sample, tag) in train_loader:
        sample = torch.as_tensor(sample)
        sample = sample.to(device)
        #print('Tag here\n', tag)
        #print('sample here\n', sample)
        #tag = np.argmax(tag)
        #print('Tag here\n',tag)
        tag = torch.as_tensor(tag)
        tag = tag.to(device)
        optimizer.zero_grad()
        # Forward pass
        y_pred = model(sample)

        print('Y_PRED', y_pred)
        print('TAG', tag)
        print('SAMPLE', sample)
        # Compute Loss
        loss = criterion(y_pred, tag)

        print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
        # Backward pass
        loss.backward()
        optimizer.step()
        train_acc += (y_pred == tag).sum().item()
print(train_acc/len(train_x))
print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
PATH = 'model.h5'
torch.save(model.state_dict(), PATH)


























# neural net
# class Network(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#         # Inputs to hidden layer linear transformation
#         self.hidden = nn.Linear((len(train_x[0])), 256)
#         # Output layer, len(train_y[0]) units - one for each digit
#         self.output = nn.Linear(256, len(train_y[0]))
#
#         # Define sigmoid activation and softmax output
#         self.sigmoid = nn.Sigmoid()
#         self.softmax = nn.Softmax(dim=1)
#
#     def forward(self, x):
#         # Pass the input tensor through each of our operations
#         x = self.hidden(x)
#         x = self.sigmoid(x)
#         x = self.output(x)
#         x = self.softmax(x)
#
#         return x
# Create the network and look at it's text representation
#model = Network().to(device)
# Build a feed-forward network