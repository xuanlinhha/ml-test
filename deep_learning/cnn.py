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

# tokenizer
nlp = spacy.load('en_core_web_lg')
tokenizer = Tokenizer(num_words=100000)
tokenizer.fit_on_texts(train_X.append(test_X))
word_index = tokenizer.word_index

train_X_sequences = tokenizer.texts_to_sequences(train_X)
test_X_sequences = tokenizer.texts_to_sequences(test_X)
train_max = max(len(sequence) for sequence in train_X_sequences)
test_max = max(len(sequence) for sequence in test_X_sequences)
MAX_SEQUENCE_LENGTH = max(train_max, test_max)

# word embedding
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
train_Y = to_categorical(train_Y, num_classes=7)
test_Y = to_categorical(test_Y, num_classes=7)

model.fit(pad_sequences(train_X_sequences,maxlen=MAX_SEQUENCE_LENGTH),train_Y,batch_size=512,epochs=10,
validation_data=(pad_sequences(test_X_sequences,maxlen=MAX_SEQUENCE_LENGTH),test_Y),shuffle=True)

predict_Y = model.predict_on_batch(pad_sequences(test_X_sequences,maxlen=MAX_SEQUENCE_LENGTH))

print("predict_Y: ", np.argmax(predict_Y,axis=-1),"\n")

# print result
# print(confusion_matrix(test_Y, predict_Y))
# print(classification_report(test_Y, predict_Y))
# print("Accuracy Score = {}".format(accuracy_score(predict_Y, test_Y) * 100))
