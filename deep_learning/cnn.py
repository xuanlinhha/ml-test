import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

import spacy
from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, LSTM, Embedding,Dropout,SpatialDropout1D,Conv1D,MaxPooling1D,GRU,BatchNormalization
from tensorflow.keras.layers import Input,Bidirectional,GlobalAveragePooling1D,GlobalMaxPooling1D,concatenate,LeakyReLU
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

# TEXT PROCESSING
train = pd.read_csv("../dataset/Sentiment-Analysis-Task/2018-Valence-oc-En-train.txt", sep='\t')
dev = pd.read_csv("../dataset/Sentiment-Analysis-Task/2018-Valence-oc-En-dev.txt", sep='\t')
train = train.drop(columns=['ID', 'Affect Dimension'])
dev = dev.drop(columns=['ID', 'Affect Dimension'])
# keep intensity num
train['Intensity Class']= [entry[0:entry.find(':')] for entry in train['Intensity Class']]
dev['Intensity Class']= [entry[0:entry.find(':')] for entry in dev['Intensity Class']]

# lowercase all text
train['Tweet'] = [entry.lower() for entry in train['Tweet']]
dev['Tweet'] = [entry.lower() for entry in dev['Tweet']]


# tokenize
train['Tweet'] = [word_tokenize(entry) for entry in train['Tweet']]
dev['Tweet'] = [word_tokenize(entry) for entry in dev['Tweet']]

# remove unimportant words & lemmatize based on the first tag
# mapping to wn
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

for index, entry in enumerate(train['Tweet']):
    final_words = []
    lemmatizer = WordNetLemmatizer()
    for word, tags in pos_tag(entry):
        if word not in stopwords.words('english') and word.isalpha():
            final_word = lemmatizer.lemmatize(word, tag_map[tags[0]])
            final_words.append(final_word)
    train.loc[index, 'Tweet_Final'] = str(final_words)

for index, entry in enumerate(dev['Tweet']):
    final_words = []
    lemmatizer = WordNetLemmatizer()
    for word, tags in pos_tag(entry):
        if word not in stopwords.words('english') and word.isalpha():
            final_word = lemmatizer.lemmatize(word, tag_map[tags[0]])
            final_words.append(final_word)
    dev.loc[index, 'Tweet_Final'] = str(final_words)

train_X = train['Tweet_Final']
train_Y = train['Intensity Class']
test_X = dev['Tweet_Final']
test_Y = dev['Intensity Class']

# Tokenizer
# The maximum number of words to be used
MAX_NB_WORDS = 50000
MAX_SEQUENCE_LENGTH = 250
EMBEDDING_DIM = 100
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(train_X.append(test_X))
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

train_X_sequences = pad_sequences(tokenizer.texts_to_sequences(train_X), maxlen=MAX_SEQUENCE_LENGTH)
test_X_sequences = pad_sequences(tokenizer.texts_to_sequences(test_X), maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of train_X_sequences tensor:', train_X_sequences.shape)
print('Shape of test_X_sequences tensor:', test_X_sequences.shape)

train_Y = pd.get_dummies(train_Y).values
test_Y = pd.get_dummies(test_Y).values
print('Shape of train_Y tensor:', train_Y.shape)
print('Shape of test_Y tensor:', test_Y.shape)
# print(train_Y[0:3])
# print(test_Y[0:3])

# word embedding
nlp = spacy.load('en_core_web_lg')
text_embedding = np.zeros((len(word_index) + 1, 300))
for word, i in word_index.items():
    text_embedding[i] = nlp(word).vector

# building model
model = Sequential()
model.add(
Embedding(input_dim=text_embedding.shape[0], output_dim=text_embedding.shape[1], weights=[text_embedding], input_length=MAX_SEQUENCE_LENGTH, trainable=False))
model.add(SpatialDropout1D(0.5))
model.add(Conv1D(filters=250, kernel_size=3,kernel_regularizer=regularizers.l2(0.00001), padding='same'))
model.add(LeakyReLU(alpha=0.2))
model.add(MaxPooling1D(pool_size=2))
model.add(Bidirectional(LSTM(7,dropout=0.5, recurrent_dropout=0.5,return_sequences=True)))
model.add(SpatialDropout1D(0.5))
model.add(Conv1D(filters=250, kernel_size=3,kernel_regularizer=regularizers.l2(0.00001), padding='same'))
model.add(LeakyReLU(alpha=0.2))
model.add(MaxPooling1D(pool_size=2))
model.add(Bidirectional(LSTM(7,dropout=0.5, recurrent_dropout=0.5,return_sequences=True)))
model.add(SpatialDropout1D(0.5))
model.add(Conv1D(filters=250, kernel_size=3,kernel_regularizer=regularizers.l2(0.00001), padding='same'))
model.add(LeakyReLU(alpha=0.2))
model.add(MaxPooling1D(pool_size=2))
model.add(Bidirectional(LSTM(7,dropout=0.5, recurrent_dropout=0.5)))
model.add(Dense(7,activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])

# fit model to data
# train
epochs = 5
batch_size = 64
history = model.fit(train_X_sequences, train_Y, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

accr = model.evaluate(test_X_sequences, test_Y)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
